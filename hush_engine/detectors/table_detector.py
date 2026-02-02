"""
Table Structure Detection for OCR'd Documents

Detects table structure from OCR bounding boxes to enable context-aware PII detection.
Uses spatial clustering to identify columns, rows, and headers.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re


@dataclass
class TableCell:
    """Represents a cell in a detected table."""
    text: str
    bbox: List[float]  # [x1, y1, x2, y2]
    row: int
    column: int
    is_header: bool = False


@dataclass
class TableColumn:
    """Represents a column in a detected table."""
    index: int
    header: Optional[str]
    x_center: float
    cells: List[TableCell]


@dataclass
class DetectedTable:
    """Represents a detected table structure."""
    columns: List[TableColumn]
    rows: List[List[TableCell]]
    header_row: Optional[List[TableCell]]

    def get_column_for_bbox(self, bbox: List[float]) -> Optional[TableColumn]:
        """Find which column a bounding box belongs to."""
        x_center = (bbox[0] + bbox[2]) / 2

        best_column = None
        min_distance = float('inf')

        for col in self.columns:
            distance = abs(x_center - col.x_center)
            if distance < min_distance:
                min_distance = distance
                best_column = col

        # Only return if reasonably close (within column width)
        if best_column and min_distance < 100:  # Tolerance in pixels
            return best_column
        return None


# Header text to entity type mapping
HEADER_ENTITY_MAP = {
    # Personal Identity
    "name": "PERSON",
    "full name": "PERSON",
    "first name": "PERSON",
    "last name": "PERSON",
    "gender": "GENDER",
    "sex": "GENDER",
    "nationality": "NRP",
    "citizenship": "NRP",
    "ethnicity": "NRP",

    # Contact Information
    "email": "EMAIL_ADDRESS",
    "e-mail": "EMAIL_ADDRESS",
    "email address": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER",
    "telephone": "PHONE_NUMBER",
    "mobile": "PHONE_NUMBER",
    "cell": "PHONE_NUMBER",
    "fax": "PHONE_NUMBER",
    "url": "URL",
    "website": "URL",
    "web": "URL",

    # Government IDs
    "ssn": "SSN",
    "social security": "SSN",
    "social security number": "SSN",
    "sin": "SSN",  # Canadian Social Insurance Number
    "passport": "PASSPORT",
    "passport number": "PASSPORT",
    "passport no": "PASSPORT",
    "driver": "DRIVERS_LICENSE",
    "driver's license": "DRIVERS_LICENSE",
    "drivers license": "DRIVERS_LICENSE",
    "driving license": "DRIVERS_LICENSE",
    "dl": "DRIVERS_LICENSE",
    "license number": "DRIVERS_LICENSE",
    "itin": "ITIN",
    "nhs": "UK_NHS",
    "nhs number": "UK_NHS",

    # Financial
    "credit card": "CREDIT_CARD",
    "card number": "CREDIT_CARD",
    "cc": "CREDIT_CARD",
    "bank": "BANK_NUMBER",
    "bank account": "BANK_NUMBER",
    "account number": "BANK_NUMBER",
    "routing": "BANK_NUMBER",
    "iban": "IBAN_CODE",
    "swift": "FINANCIAL",
    "bic": "FINANCIAL",
    "currency": "FINANCIAL",
    "amount": "FINANCIAL",
    "price": "FINANCIAL",
    "crypto": "CRYPTO",
    "bitcoin": "CRYPTO",
    "wallet": "CRYPTO",

    # Location
    "address": "LOCATION",
    "street": "LOCATION",
    "street address": "LOCATION",
    "city": "LOCATION",
    "state": "LOCATION",
    "province": "LOCATION",
    "zip": "LOCATION",
    "zip code": "LOCATION",
    "postal": "LOCATION",
    "postal code": "LOCATION",
    "country": "LOCATION",
    "location": "LOCATION",
    "coordinates": "COORDINATES",
    "gps": "COORDINATES",
    "latitude": "COORDINATES",
    "longitude": "COORDINATES",
    "lat": "COORDINATES",
    "long": "COORDINATES",
    "lng": "COORDINATES",

    # Medical
    "medical": "MEDICAL",
    "diagnosis": "MEDICAL",
    "condition": "MEDICAL",
    "medication": "MEDICAL",
    "prescription": "MEDICAL",
    "blood type": "MEDICAL",
    "health": "MEDICAL",

    # Technical / Device
    "ip": "IP_ADDRESS",
    "ip address": "IP_ADDRESS",
    "mac": "DEVICE_ID",
    "mac address": "DEVICE_ID",
    "imei": "DEVICE_ID",
    "uuid": "DEVICE_ID",
    "device": "DEVICE_ID",
    "device id": "DEVICE_ID",
    "serial": "DEVICE_ID",
    "serial number": "DEVICE_ID",
    "vin": "VEHICLE_ID",
    "vehicle": "VEHICLE_ID",
    "vehicle id": "VEHICLE_ID",

    # Document Elements
    "date": "DATE_TIME",
    "dob": "DATE_TIME",
    "date of birth": "DATE_TIME",
    "birth date": "DATE_TIME",
    "birthday": "DATE_TIME",

    # Organization
    "company": "COMPANY",
    "organization": "COMPANY",
    "employer": "COMPANY",
    "business": "COMPANY",

    # API Keys
    "aws": "AWS_ACCESS_KEY",
    "api key": "AWS_ACCESS_KEY",
    "access key": "AWS_ACCESS_KEY",
}


class TableDetector:
    """
    Detects table structure from OCR text blocks.

    Uses spatial clustering to identify columns based on x-coordinates
    and rows based on y-coordinates.
    """

    def __init__(self, column_tolerance: float = 50, row_tolerance: float = 20):
        """
        Initialize the table detector.

        Args:
            column_tolerance: Max horizontal distance (pixels) to group into same column
            row_tolerance: Max vertical distance (pixels) to group into same row
        """
        self.column_tolerance = column_tolerance
        self.row_tolerance = row_tolerance

    def detect_table(self, text_blocks: List[Dict]) -> Optional[DetectedTable]:
        """
        Detect table structure from OCR text blocks.

        Args:
            text_blocks: List of dicts with 'text' and 'bbox' keys
                        bbox format: [x1, y1, x2, y2]

        Returns:
            DetectedTable if table structure found, None otherwise
        """
        if not text_blocks or len(text_blocks) < 4:
            return None

        # Sort blocks by y-coordinate (top to bottom)
        sorted_blocks = sorted(text_blocks, key=lambda b: b['bbox'][1])

        # Cluster into rows based on y-coordinate
        rows = self._cluster_rows(sorted_blocks)

        if len(rows) < 2:  # Need at least header + 1 data row
            return None

        # Identify columns from first few rows
        columns = self._identify_columns(rows[:min(5, len(rows))])

        if len(columns) < 2:  # Need at least 2 columns for a table
            return None

        # Assign cells to columns
        table_rows = []
        for row_blocks in rows:
            row_cells = []
            for block in row_blocks:
                col_idx = self._find_column_index(block['bbox'], columns)
                cell = TableCell(
                    text=block['text'],
                    bbox=block['bbox'],
                    row=len(table_rows),
                    column=col_idx,
                    is_header=(len(table_rows) == 0)
                )
                row_cells.append(cell)
            table_rows.append(row_cells)

        # Build column objects with headers
        header_row = table_rows[0] if table_rows else []
        table_columns = []

        for col_idx, x_center in enumerate(columns):
            # Find header for this column
            header_text = None
            for cell in header_row:
                if cell.column == col_idx:
                    header_text = cell.text
                    break

            # Collect cells for this column
            col_cells = []
            for row in table_rows:
                for cell in row:
                    if cell.column == col_idx:
                        col_cells.append(cell)

            table_columns.append(TableColumn(
                index=col_idx,
                header=header_text,
                x_center=x_center,
                cells=col_cells
            ))

        return DetectedTable(
            columns=table_columns,
            rows=table_rows,
            header_row=header_row
        )

    def _cluster_rows(self, blocks: List[Dict]) -> List[List[Dict]]:
        """Cluster text blocks into rows based on y-coordinate."""
        if not blocks:
            return []

        rows = []
        current_row = [blocks[0]]
        current_y = (blocks[0]['bbox'][1] + blocks[0]['bbox'][3]) / 2

        for block in blocks[1:]:
            block_y = (block['bbox'][1] + block['bbox'][3]) / 2

            if abs(block_y - current_y) <= self.row_tolerance:
                # Same row
                current_row.append(block)
            else:
                # New row
                # Sort current row by x-coordinate (left to right)
                current_row.sort(key=lambda b: b['bbox'][0])
                rows.append(current_row)
                current_row = [block]
                current_y = block_y

        # Don't forget the last row
        if current_row:
            current_row.sort(key=lambda b: b['bbox'][0])
            rows.append(current_row)

        return rows

    def _identify_columns(self, rows: List[List[Dict]]) -> List[float]:
        """Identify column x-centers from the first few rows."""
        # Collect all x-centers
        x_centers = []
        for row in rows:
            for block in row:
                x_center = (block['bbox'][0] + block['bbox'][2]) / 2
                x_centers.append(x_center)

        if not x_centers:
            return []

        # Sort and cluster
        x_centers.sort()
        columns = []
        current_cluster = [x_centers[0]]

        for x in x_centers[1:]:
            if x - current_cluster[-1] <= self.column_tolerance:
                current_cluster.append(x)
            else:
                # New column - use mean of cluster
                columns.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [x]

        # Don't forget last cluster
        if current_cluster:
            columns.append(sum(current_cluster) / len(current_cluster))

        return columns

    def _find_column_index(self, bbox: List[float], columns: List[float]) -> int:
        """Find which column index a bbox belongs to."""
        x_center = (bbox[0] + bbox[2]) / 2

        min_dist = float('inf')
        best_idx = 0

        for idx, col_x in enumerate(columns):
            dist = abs(x_center - col_x)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        return best_idx

    def get_entity_type_for_header(self, header: str) -> Optional[str]:
        """
        Map a column header to expected entity type.

        Args:
            header: Column header text

        Returns:
            Entity type string if mapping found, None otherwise
        """
        if not header:
            return None

        # Normalize header
        normalized = header.lower().strip()

        # Remove common suffixes/prefixes
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = re.sub(r'\s*(number|no|#|:)$', '', normalized)

        # Direct lookup
        if normalized in HEADER_ENTITY_MAP:
            return HEADER_ENTITY_MAP[normalized]

        # Partial match - check if any key is contained in header
        for key, entity_type in HEADER_ENTITY_MAP.items():
            if key in normalized:
                return entity_type

        return None


class ContextAwarePIIDetector:
    """
    Enhances PII detection by using table context.

    When a table structure is detected, uses column headers to:
    1. Suggest expected entity types for each column
    2. Boost confidence when detected type matches header context
    3. Override low-confidence detections with header-based classification
    """

    def __init__(self, base_detector, table_detector: Optional[TableDetector] = None):
        """
        Initialize the context-aware detector.

        Args:
            base_detector: The underlying PIIDetector instance
            table_detector: TableDetector instance (creates default if None)
        """
        self.base_detector = base_detector
        self.table_detector = table_detector or TableDetector()

    def detect_with_context(
        self,
        text_blocks: List[Dict],
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Enhance detections using table context.

        Args:
            text_blocks: OCR text blocks with 'text' and 'bbox'
            detections: PII detections with 'text', 'bbox', 'entity_type', 'confidence'

        Returns:
            Enhanced detections with potentially corrected entity types
        """
        # Try to detect table structure
        table = self.table_detector.detect_table(text_blocks)

        if not table:
            return detections  # No table context available

        # Build column context map
        column_context = {}
        for col in table.columns:
            if col.header:
                expected_type = self.table_detector.get_entity_type_for_header(col.header)
                if expected_type:
                    column_context[col.index] = {
                        'header': col.header,
                        'expected_type': expected_type,
                        'x_center': col.x_center
                    }

        if not column_context:
            return detections  # No header context found

        # Enhance each detection
        enhanced = []
        for det in detections:
            enhanced_det = det.copy()

            # Find which column this detection belongs to
            column = table.get_column_for_bbox(det['bbox'])

            if column and column.index in column_context:
                ctx = column_context[column.index]
                expected_type = ctx['expected_type']
                current_type = det['entity_type']
                current_conf = det['confidence']

                # If detected type matches expected, boost confidence
                if current_type == expected_type:
                    enhanced_det['confidence'] = min(0.99, current_conf + 0.15)
                    enhanced_det['context_match'] = True

                # If low confidence and we have header context, consider override
                elif current_conf < 0.7:
                    # Check if types are related (both PII types)
                    enhanced_det['suggested_type'] = expected_type
                    enhanced_det['context_header'] = ctx['header']

                    # Override if very low confidence
                    if current_conf < 0.5:
                        enhanced_det['entity_type'] = expected_type
                        enhanced_det['confidence'] = 0.75
                        enhanced_det['context_override'] = True

            enhanced.append(enhanced_det)

        return enhanced

    def fill_missing_detections(
        self,
        text_blocks: List[Dict],
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Add detections for cells that were missed but have clear header context.

        Args:
            text_blocks: OCR text blocks
            detections: Existing PII detections

        Returns:
            Augmented list of detections
        """
        table = self.table_detector.detect_table(text_blocks)
        if not table:
            return detections

        # Get bboxes that already have detections
        detected_bboxes = set()
        for det in detections:
            bbox = tuple(det['bbox'])
            detected_bboxes.add(bbox)

        # Check each cell in table
        new_detections = list(detections)

        for col in table.columns:
            if not col.header:
                continue

            expected_type = self.table_detector.get_entity_type_for_header(col.header)
            if not expected_type:
                continue

            # Check cells in this column (skip header row)
            for cell in col.cells:
                if cell.is_header:
                    continue

                bbox_tuple = tuple(cell.bbox)
                if bbox_tuple not in detected_bboxes:
                    # This cell wasn't detected - add based on header context
                    if cell.text and len(cell.text.strip()) > 1:
                        new_detections.append({
                            'entity_type': expected_type,
                            'text': cell.text,
                            'confidence': 0.70,
                            'bbox': cell.bbox,
                            'context_inferred': True,
                            'context_header': col.header
                        })

        return new_detections
