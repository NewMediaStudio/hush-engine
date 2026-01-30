#!/usr/bin/env python3
"""
CLI tool for scrubbing images
Usage: python scrub_image.py <input_image> [output_image]
"""

import sys
import argparse
from pathlib import Path

from ocr import VisionOCR
from detectors import PIIDetector
from anonymizers import ImageAnonymizer
from PIL import Image


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

    # Step 2: Detect PII
    print("\nStep 2/3: Analyzing for sensitive information...")
    detector = PIIDetector()

    pii_regions = []
    pii_details = []

    for detection in detections:
        entities = detector.analyze_text(detection.text)
        if entities:
            pii_regions.append(detection.bbox)
            for entity in entities:
                pii_details.append({
                    'type': entity.entity_type,
                    'text': entity.text,
                    'confidence': entity.confidence,
                    'bbox': detection.bbox
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
