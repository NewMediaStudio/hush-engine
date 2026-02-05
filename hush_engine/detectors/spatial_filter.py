#!/usr/bin/env python3
"""
Spatial Filter for PII Detection

Uses bounding box geometry to improve precision by:
1. Identifying form labels (text ending with ':') and suppressing them as PII
2. Applying confidence penalties to detections in header/footer zones
3. Using spatial proximity to distinguish labels from values

This module leverages the OCR bounding boxes already available in the detection
pipeline to make context-aware filtering decisions.

Usage:
    from hush_engine.detectors.spatial_filter import apply_spatial_filtering, SpatialContext

    context = SpatialContext(
        page_width=612,
        page_height=792,
        all_detections=[{"text": "Name:", "bbox": (x1, y1, x2, y2)}, ...]
    )
    filtered = apply_spatial_filtering(entities, context)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import re
import logging

logger = logging.getLogger(__name__)

# Bbox format: (x1, y1, x2, y2) in PIL coordinates (pixel, top-left origin)
Bbox = Tuple[float, float, float, float]


@dataclass
class SpatialContext:
    """
    Spatial context for a page/image being processed.

    Provides page dimensions and all detected text blocks with their bboxes,
    enabling spatial analysis for false positive filtering.
    """
    page_width: float
    page_height: float
    all_detections: List[Dict[str, Any]] = field(default_factory=list)
    # Each detection: {"text": str, "bbox": (x1, y1, x2, y2), ...}

    # Configuration
    header_zone_ratio: float = 0.05  # Top 5% of page
    footer_zone_ratio: float = 0.05  # Bottom 5% of page
    zone_penalty: float = -0.20      # Confidence penalty for header/footer zones
    horizontal_proximity_px: float = 50.0  # Max horizontal gap for label-value pairs
    vertical_tolerance_px: float = 20.0    # Max vertical difference for same-line


def get_zone_penalty(bbox: Bbox, page_height: float, context: SpatialContext = None) -> float:
    """
    Calculate confidence penalty for detections in header/footer zones.

    Detections in the top or bottom 5% of a page are more likely to be
    UI elements, page numbers, or headers rather than actual PII.

    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        page_height: Height of the page in pixels
        context: Optional SpatialContext with custom zone ratios

    Returns:
        Negative confidence penalty (e.g., -0.20) or 0.0 for normal zones
    """
    if page_height <= 0:
        return 0.0

    header_ratio = context.header_zone_ratio if context else 0.05
    footer_ratio = context.footer_zone_ratio if context else 0.05
    penalty = context.zone_penalty if context else -0.20

    # Get y-center of the bbox
    y_center = (bbox[1] + bbox[3]) / 2
    y_ratio = y_center / page_height

    # Apply penalty if in header zone (top 5%)
    if y_ratio < header_ratio:
        logger.debug(f"Header zone penalty applied: y_ratio={y_ratio:.3f}")
        return penalty

    # Apply penalty if in footer zone (bottom 5%)
    if y_ratio > (1.0 - footer_ratio):
        logger.debug(f"Footer zone penalty applied: y_ratio={y_ratio:.3f}")
        return penalty

    return 0.0


def is_form_label(
    text: str,
    bbox: Bbox,
    all_detections: List[Dict[str, Any]],
    context: SpatialContext = None
) -> bool:
    """
    Detect if text is a form label rather than a PII value.

    Form labels typically:
    - End with ':' (e.g., "Name:", "SSN:", "Address:")
    - Are followed by a value to their right
    - Have another text detection horizontally adjacent

    Args:
        text: The detected text
        bbox: Bounding box of the text
        all_detections: All text detections on the page
        context: Optional SpatialContext with configuration

    Returns:
        True if this appears to be a form label, False if it might be a value
    """
    text_stripped = text.strip()

    # Must end with colon to be considered a label
    if not text_stripped.endswith(':'):
        return False

    # Common form label patterns (case-insensitive)
    label_patterns = [
        r'^name\s*:$',
        r'^first\s*name\s*:$',
        r'^last\s*name\s*:$',
        r'^full\s*name\s*:$',
        r'^address\s*:$',
        r'^street\s*:$',
        r'^city\s*:$',
        r'^state\s*:$',
        r'^zip\s*:$',
        r'^phone\s*:$',
        r'^email\s*:$',
        r'^ssn\s*:$',
        r'^dob\s*:$',
        r'^date\s*of\s*birth\s*:$',
        r'^patient\s*:$',
        r'^client\s*:$',
        r'^customer\s*:$',
        r'^applicant\s*:$',
        r'^employee\s*:$',
        r'^contact\s*:$',
        r'^to\s*:$',
        r'^from\s*:$',
        r'^cc\s*:$',
        r'^bcc\s*:$',
        r'^subject\s*:$',
    ]

    text_lower = text_stripped.lower()
    for pattern in label_patterns:
        if re.match(pattern, text_lower):
            # Strong match for known label patterns
            logger.debug(f"Form label matched: '{text_stripped}' -> pattern '{pattern}'")
            return True

    # Check for horizontally adjacent text (potential value)
    horizontal_proximity = context.horizontal_proximity_px if context else 50.0
    vertical_tolerance = context.vertical_tolerance_px if context else 20.0

    # Find any detection to the right of this one on the same line
    my_x2 = bbox[2]  # Right edge of this bbox
    my_y_center = (bbox[1] + bbox[3]) / 2

    for det in all_detections:
        det_bbox = det.get("bbox")
        det_text = det.get("text", "").strip()

        if not det_bbox or det_text == text_stripped:
            continue

        det_x1 = det_bbox[0]  # Left edge of other detection
        det_y_center = (det_bbox[1] + det_bbox[3]) / 2

        # Check if on same line (similar y-position)
        if abs(det_y_center - my_y_center) > vertical_tolerance:
            continue

        # Check if to the right and within proximity
        horizontal_gap = det_x1 - my_x2
        if 0 <= horizontal_gap <= horizontal_proximity:
            # Found adjacent text to the right - this is likely a label
            logger.debug(f"Form label with adjacent value: '{text_stripped}' -> '{det_text}'")
            return True

    return False


def is_value_near_label(
    text: str,
    bbox: Bbox,
    all_detections: List[Dict[str, Any]],
    context: SpatialContext = None
) -> bool:
    """
    Check if this detection is a value adjacent to a form label.

    Values are kept even if they would otherwise be filtered, if they
    appear immediately after a form label.

    Args:
        text: The detected text
        bbox: Bounding box of the text
        all_detections: All text detections on the page
        context: Optional SpatialContext with configuration

    Returns:
        True if this appears to be a value next to a label
    """
    horizontal_proximity = context.horizontal_proximity_px if context else 50.0
    vertical_tolerance = context.vertical_tolerance_px if context else 20.0

    my_x1 = bbox[0]  # Left edge
    my_y_center = (bbox[1] + bbox[3]) / 2

    for det in all_detections:
        det_bbox = det.get("bbox")
        det_text = det.get("text", "").strip()

        if not det_bbox:
            continue

        # Check if this detection ends with ':' (potential label)
        if not det_text.endswith(':'):
            continue

        det_x2 = det_bbox[2]  # Right edge of potential label
        det_y_center = (det_bbox[1] + det_bbox[3]) / 2

        # Check if on same line
        if abs(det_y_center - my_y_center) > vertical_tolerance:
            continue

        # Check if to the left of us and within proximity
        horizontal_gap = my_x1 - det_x2
        if 0 <= horizontal_gap <= horizontal_proximity:
            logger.debug(f"Value near label: '{text}' is adjacent to '{det_text}'")
            return True

    return False


def apply_spatial_filtering(
    entities: List[Dict[str, Any]],
    context: SpatialContext,
    entity_bboxes: Optional[Dict[int, Bbox]] = None
) -> List[Dict[str, Any]]:
    """
    Apply spatial filtering to a list of PII entities.

    Filtering includes:
    1. Suppressing form labels (text ending with ':')
    2. Applying zone penalties for header/footer detections
    3. Preserving values that are adjacent to labels

    Args:
        entities: List of detected entities with at least 'entity_type', 'text', 'confidence'
        context: SpatialContext with page dimensions and all text detections
        entity_bboxes: Optional mapping from entity index to bbox (if available)

    Returns:
        Filtered list of entities with adjusted confidences
    """
    if not context or context.page_height <= 0:
        return entities

    filtered = []

    for i, entity in enumerate(entities):
        entity_text = entity.get("text", "").strip()
        entity_type = entity.get("entity_type", "")
        confidence = entity.get("confidence", 0.5)

        # Get bbox for this entity (if available)
        bbox = None
        if entity_bboxes and i in entity_bboxes:
            bbox = entity_bboxes[i]
        elif "bbox" in entity:
            bbox = entity["bbox"]

        # Skip spatial filtering if no bbox available
        if not bbox:
            filtered.append(entity)
            continue

        # Check if this is a form label (should be suppressed)
        if entity_type in ("PERSON", "COMPANY", "LOCATION", "ADDRESS"):
            if is_form_label(entity_text, bbox, context.all_detections, context):
                logger.debug(f"Suppressed form label: '{entity_text}' as {entity_type}")
                continue

        # Apply zone penalty for header/footer
        zone_penalty = get_zone_penalty(bbox, context.page_height, context)
        if zone_penalty != 0:
            new_confidence = max(0.0, confidence + zone_penalty)
            entity = dict(entity)  # Copy to avoid mutation
            entity["confidence"] = new_confidence
            logger.debug(f"Zone penalty applied to '{entity_text}': {confidence:.2f} -> {new_confidence:.2f}")

        # Check if this value is adjacent to a label (boost/preserve)
        if is_value_near_label(entity_text, bbox, context.all_detections, context):
            # This is likely a real value - don't filter further
            entity = dict(entity) if not isinstance(entity, dict) else entity
            entity["_near_label"] = True

        filtered.append(entity)

    return filtered


def create_spatial_context(
    page_width: float,
    page_height: float,
    text_blocks: List[Dict[str, Any]]
) -> SpatialContext:
    """
    Create a SpatialContext from page dimensions and text blocks.

    Convenience function for creating context from file_router output.

    Args:
        page_width: Page width in pixels
        page_height: Page height in pixels
        text_blocks: List of {"text": str, "bbox": tuple} dicts

    Returns:
        SpatialContext ready for filtering
    """
    return SpatialContext(
        page_width=page_width,
        page_height=page_height,
        all_detections=text_blocks
    )
