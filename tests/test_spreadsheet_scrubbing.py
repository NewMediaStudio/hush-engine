#!/usr/bin/env python3
"""
Tests for spreadsheet scrubbing functionality
"""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anonymizers import SpreadsheetAnonymizer
from src.detectors import PIIDetector


def create_test_csv(output_path: str):
    """Create a test CSV with PII"""
    data = {
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John Smith', 'Alice Johnson', 'Bob Williams', 'Carol Davis', 'David Brown'],
        'email': [
            'john.smith@company.com',
            'alice.j@startup.io',
            'bob.w@enterprise.net',
            'carol.d@tech.org',
            'david.b@business.com'
        ],
        'phone': [
            '555-1234',
            '555-5678',
            '555-9012',
            '555-3456',
            '555-7890'
        ],
        'department': ['Sales', 'Engineering', 'Marketing', 'HR', 'Finance'],
        'salary': [75000, 95000, 68000, 82000, 105000]
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Created test CSV: {output_path}")
    return df


def create_test_xlsx(output_path: str):
    """Create a test XLSX with PII"""
    data = {
        'employee_id': [101, 102, 103, 104, 105],
        'full_name': ['Jane Doe', 'Mike Chen', 'Sarah Miller', 'Tom Anderson', 'Lisa Wang'],
        'personal_email': [
            'jane.doe@gmail.com',
            'mike.chen@yahoo.com',
            'sarah.m@hotmail.com',
            'tom.a@outlook.com',
            'lisa.wang@proton.me'
        ],
        'ssn': [
            '123-45-6789',
            '987-65-4321',
            '456-78-9012',
            '321-54-9876',
            '654-32-1098'
        ],
        'address': [
            '123 Main St, Boston MA',
            '456 Oak Ave, Seattle WA',
            '789 Pine Rd, Austin TX',
            '321 Elm Dr, Miami FL',
            '654 Maple Ln, Denver CO'
        ],
        'clearance_level': ['Secret', 'Top Secret', 'Confidential', 'Secret', 'Top Secret']
    }

    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Created test XLSX: {output_path}")
    return df


def test_anonymizer_consistency():
    """Test that anonymization is deterministic"""
    print("\n=== Test 1: Anonymizer Consistency ===")

    anonymizer = SpreadsheetAnonymizer()

    # Same input should always produce same output
    value = "john.doe@example.com"
    result1 = anonymizer._anonymize_value(value, "EMAIL_ADDRESS")
    result2 = anonymizer._anonymize_value(value, "EMAIL_ADDRESS")

    print(f"  Original: {value}")
    print(f"  First anonymization: {result1}")
    print(f"  Second anonymization: {result2}")

    assert result1 == result2, "Anonymization should be deterministic"
    print("  ✓ Consistent anonymization verified")


def test_entity_types():
    """Test different entity type replacements"""
    print("\n=== Test 2: Entity Type Replacements ===")

    anonymizer = SpreadsheetAnonymizer()

    test_cases = [
        ("John Smith", "PERSON"),
        ("john@example.com", "EMAIL_ADDRESS"),
        ("555-1234", "PHONE_NUMBER"),
        ("AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY"),
        ("4532-1488-0343-6467", "CREDIT_CARD"),
    ]

    for value, entity_type in test_cases:
        anonymized = anonymizer._anonymize_value(value, entity_type)
        print(f"  {entity_type}: '{value}' → '{anonymized}'")

    print("  ✓ All entity types handled")


def test_dataframe_anonymization():
    """Test full DataFrame anonymization"""
    print("\n=== Test 3: DataFrame Anonymization ===")

    # Create test DataFrame
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol'],
        'email': ['alice@test.com', 'bob@test.com', 'carol@test.com'],
        'department': ['Sales', 'Engineering', 'Marketing']
    })

    # Define entity map
    entity_map = {
        'name': ['PERSON'],
        'email': ['EMAIL_ADDRESS']
        # 'department' not in map, should remain unchanged
    }

    anonymizer = SpreadsheetAnonymizer()
    scrubbed_df = anonymizer.anonymize_dataframe(df, entity_map)

    print("  Original DataFrame:")
    print(df.to_string(index=False))
    print("\n  Scrubbed DataFrame:")
    print(scrubbed_df.to_string(index=False))

    # Verify department column unchanged
    assert scrubbed_df['department'].equals(df['department']), "Non-PII columns should be unchanged"

    # Verify name and email changed
    assert not scrubbed_df['name'].equals(df['name']), "Name column should be anonymized"
    assert not scrubbed_df['email'].equals(df['email']), "Email column should be anonymized"

    print("\n  ✓ DataFrame anonymization works correctly")


def test_column_detection():
    """Test PII detection in DataFrame columns"""
    print("\n=== Test 4: Column PII Detection ===")

    # Create test data
    csv_path = "/tmp/test_detection.csv"
    df = create_test_csv(csv_path)

    # Analyze columns
    detector = PIIDetector()

    # Manually analyze each column
    for column in df.columns:
        sample = df[column].head(3)
        sample_text = " | ".join(str(val) for val in sample)
        entities = detector.analyze_text(sample_text)

        if entities:
            entity_types = list(set([e.entity_type for e in entities]))
            print(f"  Column '{column}': {entity_types}")

    os.remove(csv_path)
    print("\n  ✓ Column detection working")


def test_csv_scrubbing():
    """Test end-to-end CSV scrubbing"""
    print("\n=== Test 5: CSV File Scrubbing ===")

    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from scrub_spreadsheet import SpreadsheetScrubber

    # Create test file
    input_path = "/tmp/test_customers.csv"
    output_path = "/tmp/test_customers_scrubbed.csv"

    original_df = create_test_csv(input_path)

    # Scrub it
    scrubber = SpreadsheetScrubber(sample_size=5, confidence_threshold=0.5)
    result = scrubber.scrub_file(input_path, output_path)

    # Load scrubbed version
    scrubbed_df = pd.read_csv(output_path)

    print(f"\n  Original data (first 2 rows):")
    print(original_df.head(2).to_string(index=False))
    print(f"\n  Scrubbed data (first 2 rows):")
    print(scrubbed_df.head(2).to_string(index=False))

    print(f"\n  Results:")
    print(f"    Rows: {result['rows']}")
    print(f"    Columns: {result['columns']}")
    print(f"    PII columns detected: {result['pii_columns']}")

    # Verify file exists
    assert Path(output_path).exists(), "Output file should exist"

    # Verify non-PII columns unchanged
    assert scrubbed_df['customer_id'].equals(original_df['customer_id']), "IDs should not change"
    assert scrubbed_df['department'].equals(original_df['department']), "Department should not change"
    assert scrubbed_df['salary'].equals(original_df['salary']), "Salary should not change"

    # Verify PII columns changed
    assert not scrubbed_df['name'].equals(original_df['name']), "Names should be anonymized"
    assert not scrubbed_df['email'].equals(original_df['email']), "Emails should be anonymized"

    # Cleanup
    os.remove(input_path)
    os.remove(output_path)

    print("  ✓ CSV scrubbing successful")


def test_xlsx_scrubbing():
    """Test end-to-end XLSX scrubbing"""
    print("\n=== Test 6: XLSX File Scrubbing ===")

    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from scrub_spreadsheet import SpreadsheetScrubber

    # Create test file
    input_path = "/tmp/test_employees.xlsx"
    output_path = "/tmp/test_employees_scrubbed.xlsx"

    original_df = create_test_xlsx(input_path)

    # Scrub it
    scrubber = SpreadsheetScrubber(sample_size=5, confidence_threshold=0.5)
    result = scrubber.scrub_file(input_path, output_path)

    # Load scrubbed version
    scrubbed_df = pd.read_excel(output_path, engine='openpyxl')

    print(f"\n  Original data (first 2 rows):")
    print(original_df.head(2).to_string(index=False))
    print(f"\n  Scrubbed data (first 2 rows):")
    print(scrubbed_df.head(2).to_string(index=False))

    print(f"\n  Results:")
    print(f"    Rows: {result['rows']}")
    print(f"    Columns: {result['columns']}")
    print(f"    PII columns detected: {result['pii_columns']}")

    # Verify file exists
    assert Path(output_path).exists(), "Output file should exist"

    # Verify non-PII columns unchanged
    assert scrubbed_df['employee_id'].equals(original_df['employee_id']), "IDs should not change"
    assert scrubbed_df['clearance_level'].equals(original_df['clearance_level']), "Clearance should not change"

    # Verify PII columns changed
    assert not scrubbed_df['full_name'].equals(original_df['full_name']), "Names should be anonymized"
    assert not scrubbed_df['personal_email'].equals(original_df['personal_email']), "Emails should be anonymized"

    # Cleanup
    os.remove(input_path)
    os.remove(output_path)

    print("  ✓ XLSX scrubbing successful")


def test_null_handling():
    """Test that null values are preserved"""
    print("\n=== Test 7: Null Value Handling ===")

    df = pd.DataFrame({
        'name': ['Alice', None, 'Bob', 'Carol', None],
        'email': ['alice@test.com', 'bob@test.com', None, 'carol@test.com', None]
    })

    entity_map = {
        'name': ['PERSON'],
        'email': ['EMAIL_ADDRESS']
    }

    anonymizer = SpreadsheetAnonymizer()
    scrubbed_df = anonymizer.anonymize_dataframe(df, entity_map)

    # Count nulls before and after
    original_nulls = df.isnull().sum().sum()
    scrubbed_nulls = scrubbed_df.isnull().sum().sum()

    print(f"  Original null count: {original_nulls}")
    print(f"  Scrubbed null count: {scrubbed_nulls}")

    assert original_nulls == scrubbed_nulls, "Null values should be preserved"

    print("  ✓ Null values preserved correctly")


def main():
    """Run all tests"""
    print("=" * 60)
    print("SPREADSHEET SCRUBBING TESTS")
    print("=" * 60)

    try:
        test_anonymizer_consistency()
        test_entity_types()
        test_dataframe_anonymization()
        test_column_detection()
        test_null_handling()
        test_csv_scrubbing()
        test_xlsx_scrubbing()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
