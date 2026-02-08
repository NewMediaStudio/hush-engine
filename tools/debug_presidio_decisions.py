#!/usr/bin/env python3
"""
Debug Presidio Detection Decisions

Uses return_decision_process=True to trace why specific detections
are being made and what context signals boost their scores.

This helps identify:
- Which recognizer fired (pattern, spaCy, custom, etc.)
- Original score vs context-boosted score
- What context words triggered the boost
- Recognition metadata (detection sources, engine counts)

Usage:
    # Run with default test cases
    python3 tools/debug_presidio_decisions.py

    # Analyze specific text
    python3 tools/debug_presidio_decisions.py --text "John Smith works at Apple"

    # Batch mode from file
    python3 tools/debug_presidio_decisions.py --file samples.txt

    # Compare raw Presidio vs Hush Engine
    python3 tools/debug_presidio_decisions.py --compare "123 Main St, New York"

    # JSON output for programmatic use
    python3 tools/debug_presidio_decisions.py --json --text "email@example.com"

    # Filter by entity type
    python3 tools/debug_presidio_decisions.py --entity PERSON --text "Dr. John Smith"
"""

import sys
import os
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# WORKAROUND: Import lightgbm first to avoid segfault on macOS
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer


@dataclass
class DecisionTrace:
    """Structured representation of a detection decision."""
    entity_type: str
    text: str
    start: int
    end: int
    final_score: float
    original_score: float
    context_boost: float
    recognizer: str
    pattern_name: Optional[str]
    supportive_context: Optional[str]
    textual_explanation: Optional[str]
    recognition_metadata: Optional[Dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)


class PresidioDebugger:
    """
    Debug analyzer that provides detailed decision traces for Presidio detections.

    Supports both raw Presidio analysis and Hush Engine analysis for comparison.
    """

    def __init__(self, use_hush_engine: bool = False, verbose: bool = True):
        """
        Initialize the debugger.

        Args:
            use_hush_engine: If True, use Hush Engine's PIIDetector instead of raw Presidio
            verbose: If True, print detailed output during analysis
        """
        self.verbose = verbose
        self.use_hush_engine = use_hush_engine

        if use_hush_engine:
            from hush_engine.detectors.pii_detector import PIIDetector
            self._detector = PIIDetector()
            self._analyzer = self._detector.analyzer
        else:
            # Configure NLP engine for context-aware detection
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()

            # Use same context enhancer settings as Hush Engine
            context_enhancer = LemmaContextAwareEnhancer(
                context_similarity_factor=0.35,
                min_score_with_context_similarity=0.4,
                context_prefix_count=5,
                context_suffix_count=5
            )

            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                context_aware_enhancer=context_enhancer
            )
            self._detector = None

    def get_registered_recognizers(self) -> List[str]:
        """Get list of all registered recognizers."""
        recognizers = self._analyzer.registry.recognizers
        return sorted(set(r.name for r in recognizers))

    def get_recognizer_by_entity(self, entity_type: str) -> List[str]:
        """Get recognizers that support a specific entity type."""
        recognizers = self._analyzer.registry.recognizers
        return sorted(set(
            r.name for r in recognizers
            if entity_type in r.supported_entities
        ))

    def analyze_with_trace(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        language: str = "en"
    ) -> List[DecisionTrace]:
        """
        Analyze text and return detailed decision traces.

        Args:
            text: Text to analyze
            entities: List of entity types to detect (None = all)
            language: Language code

        Returns:
            List of DecisionTrace objects with full decision information
        """
        results = self._analyzer.analyze(
            text=text,
            language=language,
            entities=entities,
            return_decision_process=True
        )

        traces = []
        for r in results:
            # Extract analysis explanation
            exp = getattr(r, 'analysis_explanation', None)

            original_score = r.score
            context_boost = 0.0
            recognizer_name = "unknown"
            pattern_name = None
            supportive_context = None
            textual_explanation = None

            if exp:
                recognizer_name = getattr(exp, 'recognizer', 'unknown')
                pattern_name = getattr(exp, 'pattern_name', None)
                original_score = getattr(exp, 'original_score', r.score)
                context_boost = getattr(exp, 'score_context_improvement', 0.0)
                supportive_context = getattr(exp, 'supportive_context_word', None)
                textual_explanation = getattr(exp, 'textual_explanation', None)

            # Extract recognition metadata
            recognition_metadata = getattr(r, 'recognition_metadata', None)
            if recognition_metadata:
                # Convert to dict if needed
                if hasattr(recognition_metadata, 'items'):
                    recognition_metadata = dict(recognition_metadata)

            traces.append(DecisionTrace(
                entity_type=r.entity_type,
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                final_score=r.score,
                original_score=original_score,
                context_boost=context_boost,
                recognizer=recognizer_name,
                pattern_name=pattern_name,
                supportive_context=supportive_context,
                textual_explanation=textual_explanation,
                recognition_metadata=recognition_metadata
            ))

        return sorted(traces, key=lambda t: (-t.final_score, t.start))

    def analyze_with_hush_comparison(
        self,
        text: str,
        entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare raw Presidio results with Hush Engine results.

        Shows what detections are filtered by Hush Engine's post-processing.

        Args:
            text: Text to analyze
            entities: List of entity types to detect (None = all)

        Returns:
            Dict with 'raw_presidio', 'hush_engine', 'filtered', and 'added' keys
        """
        # Get raw Presidio results
        raw_traces = self.analyze_with_trace(text, entities)

        # Get Hush Engine results
        from hush_engine.detectors.pii_detector import PIIDetector
        detector = PIIDetector()
        hush_results = detector.analyze_text(text)

        # Convert Hush results to comparable format
        hush_entities = set()
        for e in hush_results:
            hush_entities.add((e.entity_type, e.start, e.end, e.text))

        raw_entities = set()
        for t in raw_traces:
            raw_entities.add((t.entity_type, t.start, t.end, t.text))

        # Find filtered (in raw but not in Hush) and added (in Hush but not raw)
        filtered = raw_entities - hush_entities
        added = hush_entities - raw_entities

        return {
            'raw_presidio': raw_traces,
            'hush_engine': hush_results,
            'filtered_by_hush': [
                t for t in raw_traces
                if (t.entity_type, t.start, t.end, t.text) in filtered
            ],
            'added_by_hush': [
                e for e in hush_results
                if (e.entity_type, e.start, e.end, e.text) in added
            ]
        }

    def group_by_recognizer(
        self,
        traces: List[DecisionTrace]
    ) -> Dict[str, List[DecisionTrace]]:
        """Group traces by recognizer name."""
        groups = {}
        for t in traces:
            if t.recognizer not in groups:
                groups[t.recognizer] = []
            groups[t.recognizer].append(t)
        return groups

    def group_by_entity_type(
        self,
        traces: List[DecisionTrace]
    ) -> Dict[str, List[DecisionTrace]]:
        """Group traces by entity type."""
        groups = {}
        for t in traces:
            if t.entity_type not in groups:
                groups[t.entity_type] = []
            groups[t.entity_type].append(t)
        return groups

    def find_context_boosted(
        self,
        traces: List[DecisionTrace],
        min_boost: float = 0.05
    ) -> List[DecisionTrace]:
        """Find traces that received context boost."""
        return [t for t in traces if t.context_boost >= min_boost]

    def print_decision_trace(
        self,
        traces: List[DecisionTrace],
        show_metadata: bool = True,
        show_context: bool = True
    ):
        """Print detailed decision trace for each detection."""
        if not traces:
            print("  No detections found.")
            return

        for t in traces:
            print(f"\n{'='*60}")
            print(f"Entity Type: {t.entity_type}")
            print(f"Text: \"{t.text}\"")
            print(f"Position: {t.start}:{t.end}")
            print(f"Final Score: {t.final_score:.3f}")

            print(f"\nDecision Details:")
            print(f"  Recognizer: {t.recognizer}")
            if t.pattern_name:
                print(f"  Pattern Name: {t.pattern_name}")
            print(f"  Original Score: {t.original_score:.3f}")

            if show_context and t.context_boost > 0:
                print(f"  Context Boost: +{t.context_boost:.3f}")
                if t.supportive_context:
                    print(f"  Boosting Context Word: \"{t.supportive_context}\"")

            if t.textual_explanation:
                print(f"  Explanation: {t.textual_explanation}")

            if show_metadata and t.recognition_metadata:
                print(f"\nRecognition Metadata:")
                for key, value in t.recognition_metadata.items():
                    print(f"  {key}: {value}")

    def print_recognizer_breakdown(self, traces: List[DecisionTrace]):
        """Print traces grouped by recognizer."""
        groups = self.group_by_recognizer(traces)

        print(f"\n{'='*70}")
        print("BREAKDOWN BY RECOGNIZER")
        print('='*70)

        for recognizer, rec_traces in sorted(groups.items()):
            print(f"\n[{recognizer}] ({len(rec_traces)} detections)")
            print("-" * 50)
            for t in rec_traces:
                boost_str = f" (+{t.context_boost:.2f} context)" if t.context_boost > 0 else ""
                ctx_str = f" <- \"{t.supportive_context}\"" if t.supportive_context else ""
                print(f"  {t.entity_type}: \"{t.text}\" (score: {t.final_score:.2f}{boost_str}{ctx_str})")

    def print_context_analysis(self, traces: List[DecisionTrace]):
        """Print analysis of context-boosted detections."""
        boosted = self.find_context_boosted(traces)

        print(f"\n{'='*70}")
        print("CONTEXT BOOST ANALYSIS")
        print('='*70)

        if not boosted:
            print("  No detections received context boost.")
            return

        print(f"\n{len(boosted)} detection(s) received context boost:\n")
        for t in boosted:
            print(f"  {t.entity_type}: \"{t.text}\"")
            print(f"    Original: {t.original_score:.3f} -> Final: {t.final_score:.3f} (+{t.context_boost:.3f})")
            if t.supportive_context:
                print(f"    Context word: \"{t.supportive_context}\"")
            print()


def debug_sample(
    debugger: PresidioDebugger,
    text: str,
    entities: Optional[List[str]] = None,
    show_breakdown: bool = True,
    show_context: bool = True
) -> List[DecisionTrace]:
    """Debug a specific text sample with full output."""
    print(f"\n{'#'*70}")
    print("INPUT TEXT:")
    print(f"  \"{text[:200]}{'...' if len(text) > 200 else ''}\"")
    print('#'*70)

    traces = debugger.analyze_with_trace(text, entities)

    debugger.print_decision_trace(traces)

    if show_breakdown:
        debugger.print_recognizer_breakdown(traces)

    if show_context:
        debugger.print_context_analysis(traces)

    return traces


def compare_sample(debugger: PresidioDebugger, text: str) -> Dict[str, Any]:
    """Compare raw Presidio vs Hush Engine results."""
    print(f"\n{'#'*70}")
    print("COMPARISON: Raw Presidio vs Hush Engine")
    print(f"INPUT: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
    print('#'*70)

    comparison = debugger.analyze_with_hush_comparison(text)

    print(f"\n[Raw Presidio] {len(comparison['raw_presidio'])} detection(s)")
    for t in comparison['raw_presidio']:
        print(f"  {t.entity_type}: \"{t.text}\" ({t.recognizer}, score: {t.final_score:.2f})")

    print(f"\n[Hush Engine] {len(comparison['hush_engine'])} detection(s)")
    for e in comparison['hush_engine']:
        print(f"  {e.entity_type}: \"{e.text}\" (score: {e.confidence:.2f})")

    if comparison['filtered_by_hush']:
        print(f"\n[FILTERED by Hush Engine] {len(comparison['filtered_by_hush'])} detection(s)")
        for t in comparison['filtered_by_hush']:
            print(f"  {t.entity_type}: \"{t.text}\" (was: {t.recognizer}, score: {t.final_score:.2f})")

    if comparison['added_by_hush']:
        print(f"\n[ADDED by Hush Engine] {len(comparison['added_by_hush'])} detection(s)")
        for e in comparison['added_by_hush']:
            pattern = getattr(e, 'pattern_name', None) or 'custom'
            print(f"  {e.entity_type}: \"{e.text}\" (pattern: {pattern}, score: {e.confidence:.2f})")

    return comparison


# Comprehensive test samples for common false positive scenarios
FALSE_POSITIVE_SCENARIOS = [
    # Form Labels
    {
        "text": "Demographics: Age 45, Gender: Male, Race: Caucasian",
        "desc": "Form labels detected as PERSON/LOCATION",
        "expected_fp": ["Race", "Gender", "Male", "Caucasian"]
    },
    {
        "text": "Contact Information: Name: Address: Phone:",
        "desc": "Empty form labels",
        "expected_fp": ["Name", "Address", "Phone"]
    },
    {
        "text": "First Name: Last Name: Email: Company:",
        "desc": "Labeled fields without values",
        "expected_fp": ["First Name", "Last Name", "Email", "Company"]
    },

    # Business/UI Text
    {
        "text": "The customer portal allows agents to view requests and manage support tickets.",
        "desc": "UI navigation text",
        "expected_fp": ["customer", "portal", "agents"]
    },
    {
        "text": "Customer Hub | Overview | Dashboard | New Customer",
        "desc": "Navigation menu items",
        "expected_fp": ["Customer Hub", "Overview", "Dashboard"]
    },

    # Company Names
    {
        "text": "The patient visited Apple headquarters in California.",
        "desc": "Company name in sentence",
        "expected_fp": []
    },
    {
        "text": "Microsoft Azure and Amazon AWS provide cloud services.",
        "desc": "Multiple company names",
        "expected_fp": []
    },

    # Addresses
    {
        "text": "Order #12345 shipped to 123 Main Street, New York, NY 10001",
        "desc": "Full US address with order number",
        "expected_fp": []
    },
    {
        "text": "Toronto, ON N0H 1K0",
        "desc": "Canadian city/province/postal",
        "expected_fp": []
    },
    {
        "text": "123 Rue Saint-Denis, Montreal, QC H2X 1K3",
        "desc": "French-Canadian street address",
        "expected_fp": []
    },

    # Person Names
    {
        "text": "Dr. John Smith visited St. Patrick's Hospital.",
        "desc": "Person name with title",
        "expected_fp": []
    },
    {
        "text": "The report was authored by Smith, John A. and reviewed by Jane Doe.",
        "desc": "Inverted name format",
        "expected_fp": []
    },

    # Technical/Credentials
    {
        "text": "API key: sk_live_abc123def456ghi789",
        "desc": "API key credential",
        "expected_fp": []
    },
    {
        "text": "Database password: CorrectHorse42Battery!Staple",
        "desc": "Password-like string",
        "expected_fp": []
    },

    # Dates
    {
        "text": "Meeting scheduled for 12/25/2024 at 2:30 PM",
        "desc": "US date format",
        "expected_fp": []
    },
    {
        "text": "DOB: 1990-05-15, Admission: 25-Dec-2023",
        "desc": "Multiple date formats",
        "expected_fp": []
    },

    # Financial
    {
        "text": "Total: $1,234.56 USD, Account: 4532 1234 5678 9010",
        "desc": "Currency and credit card",
        "expected_fp": []
    },

    # Phone Numbers
    {
        "text": "Call us at (555) 123-4567 or 1-800-EXAMPLE",
        "desc": "US phone formats",
        "expected_fp": []
    },
    {
        "text": "International: +44 20 7946 0958",
        "desc": "UK phone number",
        "expected_fp": []
    },

    # Medical
    {
        "text": "Patient diagnosed with Type 2 Diabetes, prescribed Metformin 500mg",
        "desc": "Medical condition and medication",
        "expected_fp": []
    },

    # Mixed/Complex
    {
        "text": "Invoice #INV-2024-0001 for John Smith, 123 Main St, $500.00",
        "desc": "Invoice with multiple PII types",
        "expected_fp": []
    },
    {
        "text": "Email: support@company.com | Phone: 555-1234 | Fax: 555-5678",
        "desc": "Contact block with labels",
        "expected_fp": []
    },
]


def run_batch_analysis(debugger: PresidioDebugger, samples: List[Dict]):
    """Run batch analysis on multiple samples."""
    results = []

    for sample in samples:
        text = sample.get("text", "")
        desc = sample.get("desc", "")
        expected_fp = sample.get("expected_fp", [])

        print(f"\n{'*'*70}")
        print(f"SCENARIO: {desc}")

        traces = debug_sample(debugger, text, show_breakdown=False, show_context=False)

        # Check for expected false positives
        detected_texts = [t.text for t in traces]
        unexpected_detections = []
        for t in traces:
            if t.text in expected_fp:
                print(f"  [WARNING] Detected expected FP: {t.entity_type} \"{t.text}\"")

        results.append({
            "text": text,
            "desc": desc,
            "traces": [t.to_dict() for t in traces],
            "detection_count": len(traces)
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Debug Presidio detection decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/debug_presidio_decisions.py
  python3 tools/debug_presidio_decisions.py --text "John Smith at 123 Main St"
  python3 tools/debug_presidio_decisions.py --compare "Dr. Jane Doe"
  python3 tools/debug_presidio_decisions.py --file samples.txt --json
  python3 tools/debug_presidio_decisions.py --entity PERSON --text "Bob Jones"
  python3 tools/debug_presidio_decisions.py --list-recognizers
        """
    )

    parser.add_argument(
        "--text", "-t",
        help="Text to analyze"
    )
    parser.add_argument(
        "--file", "-f",
        help="File containing text samples (one per line)"
    )
    parser.add_argument(
        "--compare", "-c",
        help="Compare raw Presidio vs Hush Engine for given text"
    )
    parser.add_argument(
        "--entity", "-e",
        help="Filter by entity type (e.g., PERSON, LOCATION)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--hush",
        action="store_true",
        help="Use Hush Engine instead of raw Presidio"
    )
    parser.add_argument(
        "--list-recognizers",
        action="store_true",
        help="List all registered recognizers"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive mode"
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all false positive test scenarios"
    )

    args = parser.parse_args()

    # Initialize debugger
    debugger = PresidioDebugger(use_hush_engine=args.hush, verbose=not args.json)

    # List recognizers
    if args.list_recognizers:
        print("Registered Recognizers:")
        for name in debugger.get_registered_recognizers():
            print(f"  - {name}")
        return

    # Entity filter
    entities = [args.entity] if args.entity else None

    # JSON output mode
    all_results = []

    # Compare mode
    if args.compare:
        result = compare_sample(debugger, args.compare)
        if args.json:
            print(json.dumps({
                "text": args.compare,
                "raw_presidio": [t.to_dict() for t in result['raw_presidio']],
                "hush_filtered": [t.to_dict() for t in result['filtered_by_hush']],
            }, indent=2))
        return

    # Single text mode
    if args.text:
        traces = debug_sample(debugger, args.text, entities)
        if args.json:
            print(json.dumps([t.to_dict() for t in traces], indent=2))
        return

    # File mode
    if args.file:
        with open(args.file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            traces = debug_sample(debugger, line, entities)
            all_results.append({
                "text": line,
                "traces": [t.to_dict() for t in traces]
            })

        if args.json:
            print(json.dumps(all_results, indent=2))
        return

    # All scenarios mode
    if args.all_scenarios:
        results = run_batch_analysis(debugger, FALSE_POSITIVE_SCENARIOS)
        if args.json:
            print(json.dumps(results, indent=2))
        return

    # Default: Run built-in test samples
    if not args.json:
        print("=" * 70)
        print("PRESIDIO DECISION PROCESS DEBUGGER")
        print("=" * 70)
        print("Uses return_decision_process=True to trace detection decisions")
        print("Shows original_score, context boost, and which recognizer fired")

        if args.hush:
            print("\n[Using Hush Engine with custom recognizers]")
        else:
            print("\n[Using raw Presidio with default recognizers]")

    # Run sample scenarios
    sample_scenarios = FALSE_POSITIVE_SCENARIOS[:5]  # Just first 5 by default
    for sample in sample_scenarios:
        if not args.json:
            print(f"\n\n{'*'*70}")
            print(f"SCENARIO: {sample['desc']}")
        traces = debug_sample(debugger, sample["text"], entities)
        all_results.append({
            "text": sample["text"],
            "desc": sample["desc"],
            "traces": [t.to_dict() for t in traces]
        })

    if args.json:
        print(json.dumps(all_results, indent=2))
        return

    # Interactive mode
    if not args.no_interactive:
        print("\n\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("Enter text to analyze (or 'quit' to exit):")
        print("Commands: 'compare <text>' to compare Presidio vs Hush")
        print("          'recognizers' to list all recognizers")
        print("          'recognizers <entity>' to list recognizers for entity type")
        print("=" * 70)

        while True:
            try:
                text = input("\n> ").strip()
                if text.lower() in ('quit', 'exit', 'q'):
                    break

                if text.lower() == 'recognizers':
                    print("\nRegistered Recognizers:")
                    for name in debugger.get_registered_recognizers():
                        print(f"  - {name}")
                    continue

                if text.lower().startswith('recognizers '):
                    entity = text.split(' ', 1)[1].upper()
                    print(f"\nRecognizers for {entity}:")
                    for name in debugger.get_recognizer_by_entity(entity):
                        print(f"  - {name}")
                    continue

                if text.lower().startswith('compare '):
                    compare_text = text.split(' ', 1)[1]
                    compare_sample(debugger, compare_text)
                    continue

                if text:
                    debug_sample(debugger, text, entities)

            except (EOFError, KeyboardInterrupt):
                break

    if not args.json:
        print("\nDone.")


if __name__ == "__main__":
    main()
