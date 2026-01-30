#!/usr/bin/env python3
"""
Unit tests for Vision OCR
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr import VisionOCR, TextDetection
from PIL import Image, ImageDraw, ImageFont


def create_test_image(text: str, output_path: str, size=(800, 200)):
    """Create a simple test image with text"""
    # Create white background
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw text (using default font)
    # Position text in center
    text_position = (50, 80)
    draw.text(text_position, text, fill='black')

    img.save(output_path)
    print(f"Created test image: {output_path}")


def test_basic_ocr():
    """Test basic text extraction"""
    print("\n=== Test 1: Basic OCR ===")

    # Create test image
    test_image_path = "/tmp/test_ocr_basic.png"
    test_text = "Hello World 123"
    create_test_image(test_text, test_image_path)

    # Run OCR
    ocr = VisionOCR(recognition_level="accurate")
    detections = ocr.extract_text(test_image_path)

    print(f"Detected {len(detections)} text regions:")
    for i, det in enumerate(detections):
        print(f"  [{i}] Text: '{det.text}'")
        print(f"      Confidence: {det.confidence:.2f}")
        print(f"      BBox: {det.bbox}")

    # Cleanup
    os.remove(test_image_path)

    assert len(detections) > 0, "Should detect at least one text region"
    print("✓ Basic OCR test passed")


def test_pii_detection_image():
    """Test with PII-like content"""
    print("\n=== Test 2: PII Content OCR ===")

    # Create test image with fake PII
    test_image_path = "/tmp/test_ocr_pii.png"
    test_text = "Email: john.doe@example.com\nPhone: 555-1234"
    create_test_image(test_text, test_image_path, size=(800, 300))

    # Run OCR
    ocr = VisionOCR(recognition_level="accurate")
    detections = ocr.extract_text(test_image_path)

    print(f"Detected {len(detections)} text regions:")
    for i, det in enumerate(detections):
        print(f"  [{i}] '{det.text}' (conf: {det.confidence:.2f})")

    # Cleanup
    os.remove(test_image_path)

    assert len(detections) > 0, "Should detect text"
    print("✓ PII content OCR test passed")


def test_coordinate_transformation():
    """Test coordinate transformation logic"""
    print("\n=== Test 3: Coordinate Transformation ===")

    # Test with known values
    # Vision box at bottom-left corner: (0, 0, 0.5, 0.1)
    # Image size: 1000x500
    vision_box = (0.0, 0.0, 0.5, 0.1)
    img_height = 500
    img_width = 1000

    pil_coords = VisionOCR.vision_to_pil_coords(
        vision_box,
        img_height,
        img_width
    )

    print(f"Vision box (normalized, bottom-left): {vision_box}")
    print(f"PIL coords (pixels, top-left): {pil_coords}")

    # In Vision: (0, 0, 0.5, 0.1) means:
    #   - Start at bottom-left (0, 0)
    #   - Width: 50% of image = 500px
    #   - Height: 10% of image = 50px
    # In PIL (top-left origin):
    #   - y should be: (1 - 0 - 0.1) * 500 = 450
    #   - x should be: 0
    #   - x2 should be: 500
    #   - y2 should be: 500

    expected = (0, 450, 500, 500)
    assert pil_coords == expected, f"Expected {expected}, got {pil_coords}"

    print("✓ Coordinate transformation test passed")


def test_fast_vs_accurate():
    """Compare fast vs accurate recognition modes"""
    print("\n=== Test 4: Fast vs Accurate Mode ===")

    test_image_path = "/tmp/test_ocr_modes.png"
    test_text = "Quick performance test"
    create_test_image(test_text, test_image_path)

    import time

    # Test fast mode
    ocr_fast = VisionOCR(recognition_level="fast")
    start = time.time()
    detections_fast = ocr_fast.extract_text(test_image_path)
    time_fast = time.time() - start

    # Test accurate mode
    ocr_accurate = VisionOCR(recognition_level="accurate")
    start = time.time()
    detections_accurate = ocr_accurate.extract_text(test_image_path)
    time_accurate = time.time() - start

    print(f"Fast mode: {len(detections_fast)} detections in {time_fast:.3f}s")
    print(f"Accurate mode: {len(detections_accurate)} detections in {time_accurate:.3f}s")

    # Cleanup
    os.remove(test_image_path)

    print("✓ Mode comparison test passed")


def main():
    """Run all tests"""
    print("Testing Vision OCR Implementation")
    print("=" * 50)

    try:
        test_coordinate_transformation()
        test_basic_ocr()
        test_pii_detection_image()
        test_fast_vs_accurate()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
