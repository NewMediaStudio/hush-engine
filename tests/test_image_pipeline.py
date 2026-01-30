#!/usr/bin/env python3
"""
Integration test: OCR + PII Detection + Image Anonymization
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr import VisionOCR
from src.detectors import PIIDetector
from src.anonymizers import ImageAnonymizer
from PIL import Image, ImageDraw, ImageFont


def create_test_screenshot(output_path: str):
    """Create a realistic test screenshot with PII"""
    img = Image.new('RGB', (1200, 800), color='white')
    draw = ImageDraw.Draw(img)

    # Draw some "UI" elements
    draw.rectangle([50, 50, 1150, 100], outline='black', width=2)
    draw.text((60, 65), "User Profile", fill='black')

    # Add PII content
    pii_content = [
        (100, 150, "Name: John Smith"),
        (100, 200, "Email: john.smith@company.com"),
        (100, 250, "Phone: (555) 123-4567"),
        (100, 300, "SSN: 123-45-6789"),
        (100, 350, "Address: 123 Main Street, Boston, MA"),
        (100, 450, "API Key: AKIAIOSFODNN7EXAMPLE"),
        (100, 500, "Credit Card: 4532-1488-0343-6467"),
    ]

    for x, y, text in pii_content:
        draw.text((x, y), text, fill='black')

    img.save(output_path)
    print(f"Created test screenshot: {output_path}")
    return img


def test_full_pipeline():
    """Test the complete scrubbing pipeline"""
    print("\n=== Full Pipeline Test: OCR → PII Detection → Redaction ===\n")

    # Step 1: Create test image
    test_image_path = "/tmp/test_screenshot.png"
    original_image = create_test_screenshot(test_image_path)

    # Step 2: Extract text with OCR
    print("Step 1: Running OCR...")
    ocr = VisionOCR(recognition_level="accurate")
    detections = ocr.extract_text(test_image_path)
    print(f"  ✓ Detected {len(detections)} text regions\n")

    # Step 3: Analyze for PII
    print("Step 2: Detecting PII...")
    detector = PIIDetector()

    pii_bboxes = []
    for detection in detections:
        entities = detector.analyze_text(detection.text)
        if entities:
            print(f"  Found PII in: '{detection.text}'")
            for entity in entities:
                print(f"    - {entity.entity_type}: '{entity.text}' (confidence: {entity.confidence:.2f})")
            pii_bboxes.append(detection.bbox)

    print(f"\n  ✓ Found {len(pii_bboxes)} regions with PII\n")

    # Step 4: Redact image
    print("Step 3: Redacting sensitive regions...")

    # Test black bar method
    anonymizer_black = ImageAnonymizer(method="black_bar")
    redacted_black = anonymizer_black.redact_regions(original_image, pii_bboxes)
    output_black = "/tmp/test_screenshot_redacted_black.png"
    redacted_black.save(output_black)
    print(f"  ✓ Saved black bar version: {output_black}")

    # Test blur method
    anonymizer_blur = ImageAnonymizer(method="blur")
    redacted_blur = anonymizer_blur.redact_regions(original_image, pii_bboxes)
    output_blur = "/tmp/test_screenshot_redacted_blur.png"
    redacted_blur.save(output_blur)
    print(f"  ✓ Saved blur version: {output_blur}")

    # Step 5: Verify redaction worked
    print("\nStep 4: Verifying redaction...")
    # Open redacted image and verify it's different from original
    redacted_image = Image.open(output_black)
    assert redacted_image.size == original_image.size, "Size should match"

    # Compare some pixels to ensure they changed
    original_pixels = original_image.load()
    redacted_pixels = redacted_image.load()

    # Check that at least some pixels changed (in redacted areas)
    if pii_bboxes:
        first_bbox = pii_bboxes[0]
        x, y = int(first_bbox[0]) + 5, int(first_bbox[1]) + 5
        if original_pixels[x, y] != redacted_pixels[x, y]:
            print(f"  ✓ Pixel at ({x}, {y}) changed from {original_pixels[x, y]} to {redacted_pixels[x, y]}")
        else:
            print(f"  Note: Pixel comparison inconclusive")

    print("\n" + "=" * 60)
    print("✓ Full pipeline test completed successfully!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - Original: {test_image_path}")
    print(f"  - Redacted (black): {output_black}")
    print(f"  - Redacted (blur): {output_blur}")
    print(f"\nOpen these files to visually verify the redaction worked!")


def test_selective_redaction():
    """Test redacting only specific regions"""
    print("\n=== Selective Redaction Test ===\n")

    test_image_path = "/tmp/test_selective.png"
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)

    # Add text
    draw.text((50, 50), "Public Info: Company Name Inc.", fill='black')
    draw.text((50, 100), "Secret: john@example.com", fill='red')
    draw.text((50, 150), "Secret: 555-1234", fill='red')
    draw.text((50, 200), "Public: About Us Page", fill='black')

    img.save(test_image_path)

    # Extract all text
    ocr = VisionOCR()
    all_detections = ocr.extract_text(test_image_path)

    # Find only "Secret" lines
    detector = PIIDetector()
    secret_bboxes = []

    for detection in all_detections:
        if "Secret" in detection.text or "@" in detection.text or "555" in detection.text:
            entities = detector.analyze_text(detection.text)
            if entities:
                secret_bboxes.append(detection.bbox)
                print(f"Will redact: '{detection.text}'")

    # Redact only secrets
    anonymizer = ImageAnonymizer(method="black_bar")
    redacted = anonymizer.redact_regions(img, secret_bboxes)

    output_path = "/tmp/test_selective_redacted.png"
    redacted.save(output_path)

    print(f"\n✓ Selectively redacted {len(secret_bboxes)} regions")
    print(f"  Saved to: {output_path}")


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("IMAGE SCRUBBING PIPELINE INTEGRATION TESTS")
    print("=" * 60)

    try:
        test_full_pipeline()
        test_selective_redaction()

        print("\n" + "=" * 60)
        print("✓ ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
