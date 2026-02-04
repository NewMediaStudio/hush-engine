#!/usr/bin/env python3
"""
CLI tool for scrubbing images
Usage: python scrub_image.py <input_image> [output_image]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ocr import VisionOCR
from ocr.vision_ocr import TextDetection
from detectors import PIIDetector
from anonymizers import ImageAnonymizer
from PIL import Image


@dataclass
class MergedTextRegion:
    """A merged text region combining adjacent OCR detections for better PII detection."""
    text: str
    detections: List[TextDetection]  # Original detections that were merged
    char_offsets: List[Tuple[int, int]]  # (start, end) offset in merged text for each detection


def merge_adjacent_detections(
    detections: List[TextDetection],
    horizontal_threshold: float = 50.0,
    vertical_threshold: float = 20.0
) -> List[MergedTextRegion]:
    """
    Merge horizontally adjacent text detections into logical groups.

    This helps detect PII patterns that span multiple OCR regions, like:
    - "808921738 RT0001" (business number + program ID)
    - "John Smith" (first name + last name as separate detections)

    Args:
        detections: List of OCR text detections
        horizontal_threshold: Max horizontal gap (pixels) to consider adjacent
        vertical_threshold: Max vertical difference (pixels) to consider same line

    Returns:
        List of MergedTextRegion objects
    """
    if not detections:
        return []

    # Sort by y (top to bottom), then x (left to right)
    sorted_detections = sorted(detections, key=lambda d: (d.bbox[1], d.bbox[0]))

    merged_regions = []
    current_group = [sorted_detections[0]]

    for detection in sorted_detections[1:]:
        last_detection = current_group[-1]

        # Check if on same line (vertical alignment)
        last_y_center = (last_detection.bbox[1] + last_detection.bbox[3]) / 2
        curr_y_center = (detection.bbox[1] + detection.bbox[3]) / 2
        vertical_diff = abs(curr_y_center - last_y_center)

        # Check horizontal gap
        horizontal_gap = detection.bbox[0] - last_detection.bbox[2]

        # Merge if on same line and close enough horizontally
        if vertical_diff <= vertical_threshold and 0 <= horizontal_gap <= horizontal_threshold:
            current_group.append(detection)
        else:
            # Save current group and start new one
            merged_regions.append(_create_merged_region(current_group))
            current_group = [detection]

    # Don't forget the last group
    if current_group:
        merged_regions.append(_create_merged_region(current_group))

    return merged_regions


def _create_merged_region(detections: List[TextDetection]) -> MergedTextRegion:
    """Create a MergedTextRegion from a list of adjacent detections."""
    texts = []
    char_offsets = []
    current_offset = 0

    for i, detection in enumerate(detections):
        start = current_offset
        end = start + len(detection.text)
        char_offsets.append((start, end))
        texts.append(detection.text)
        current_offset = end + 1  # +1 for space separator

    merged_text = " ".join(texts)

    return MergedTextRegion(
        text=merged_text,
        detections=detections,
        char_offsets=char_offsets
    )


def get_bbox_for_entity(
    entity_start: int,
    entity_end: int,
    merged_region: MergedTextRegion
) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box for an entity within a merged text region.

    Args:
        entity_start: Start offset of entity in merged text
        entity_end: End offset of entity in merged text
        merged_region: The merged region containing the entity

    Returns:
        (x1, y1, x2, y2) bounding box encompassing the entity
    """
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for i, (det_start, det_end) in enumerate(merged_region.char_offsets):
        detection = merged_region.detections[i]

        # Check if this detection overlaps with the entity
        if det_start < entity_end and det_end > entity_start:
            # Calculate what portion of this detection is part of the entity
            local_start = max(0, entity_start - det_start)
            local_end = min(len(detection.text), entity_end - det_start)

            # Get bounding box for this portion
            if detection.char_boxes and len(detection.char_boxes) >= local_end:
                # Use character-level boxes for precision
                for j in range(local_start, local_end):
                    if j < len(detection.char_boxes) and detection.char_boxes[j]:
                        box = detection.char_boxes[j]
                        min_x = min(min_x, box[0])
                        min_y = min(min_y, box[1])
                        max_x = max(max_x, box[2])
                        max_y = max(max_y, box[3])
            else:
                # Fall back to detection's full bbox
                min_x = min(min_x, detection.bbox[0])
                min_y = min(min_y, detection.bbox[1])
                max_x = max(max_x, detection.bbox[2])
                max_y = max(max_y, detection.bbox[3])

    # If no valid boxes found, return a default
    if min_x == float('inf'):
        # Use the first relevant detection's bbox as fallback
        return merged_region.detections[0].bbox

    return (min_x, min_y, max_x, max_y)


def scrub_image(
    input_path: str,
    output_path: str = None,
    method: str = "black_bar",
    recognition_level: str = "accurate"
) -> dict:
    """
    Scrub an image by detecting and redacting PII

    Args:
        input_path: Path to input image
        output_path: Path to save scrubbed image (default: adds _scrubbed suffix)
        method: Redaction method ("black_bar" or "blur")
        recognition_level: OCR level ("fast" or "accurate")

    Returns:
        Dictionary with results
    """
    # Set default output path
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_scrubbed{input_file.suffix}")

    print(f"Scrubbing image: {input_path}")
    print(f"Using {method} redaction with {recognition_level} OCR\n")

    # Step 1: Extract text with OCR
    print("Step 1/3: Extracting text with OCR...")
    ocr = VisionOCR(recognition_level=recognition_level)
    detections = ocr.extract_text(input_path)
    print(f"  → Detected {len(detections)} text regions")

    # Step 1.5: Merge adjacent text regions for better PII detection
    # This helps detect patterns that span multiple OCR regions (e.g., "808921738 RT0001")
    merged_regions = merge_adjacent_detections(detections)
    print(f"  → Merged into {len(merged_regions)} logical regions")

    # Step 2: Detect PII on merged regions
    print("\nStep 2/3: Analyzing for sensitive information...")
    detector = PIIDetector()

    pii_regions = []
    pii_details = []

    for merged_region in merged_regions:
        entities = detector.analyze_text(merged_region.text)
        if entities:
            for entity in entities:
                # Calculate bounding box spanning the relevant original detections
                entity_bbox = get_bbox_for_entity(
                    entity_start=entity.start,
                    entity_end=entity.end,
                    merged_region=merged_region
                )
                pii_regions.append(entity_bbox)
                pii_details.append({
                    'type': entity.entity_type,
                    'text': entity.text,
                    'confidence': entity.confidence,
                    'bbox': entity_bbox
                })
                print(f"  → Found {entity.entity_type}: '{entity.text}' (confidence: {entity.confidence:.2f})")

    print(f"\n  → Total: {len(pii_regions)} regions with PII")

    # Step 3: Redact
    print(f"\nStep 3/3: Redacting with {method} method...")
    if pii_regions:
        original_image = Image.open(input_path)
        anonymizer = ImageAnonymizer(method=method)
        scrubbed_image = anonymizer.redact_regions(original_image, pii_regions)
        scrubbed_image.save(output_path)
        print(f"  → Saved scrubbed image to: {output_path}")
    else:
        print("  → No PII detected, no redaction needed")
        # Just copy the original
        Image.open(input_path).save(output_path)

    return {
        'input': input_path,
        'output': output_path,
        'text_regions': len(detections),
        'pii_regions': len(pii_regions),
        'pii_details': pii_details
    }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Scrub PII from images using local OCR and detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrub an image with default settings
  python scrub_image.py screenshot.png

  # Specify output file and use blur instead of black bars
  python scrub_image.py input.png output.png --method blur

  # Use fast OCR for quicker processing
  python scrub_image.py image.jpg --ocr-level fast
        """
    )

    parser.add_argument('input', help='Input image file')
    parser.add_argument('output', nargs='?', help='Output image file (default: input_scrubbed.ext)')
    parser.add_argument(
        '--method',
        choices=['black_bar', 'blur'],
        default='black_bar',
        help='Redaction method (default: black_bar)'
    )
    parser.add_argument(
        '--ocr-level',
        choices=['fast', 'accurate'],
        default='accurate',
        help='OCR recognition level (default: accurate)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        result = scrub_image(
            args.input,
            args.output,
            method=args.method,
            recognition_level=args.ocr_level
        )

        print("\n" + "=" * 60)
        print("✓ Scrubbing complete!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
