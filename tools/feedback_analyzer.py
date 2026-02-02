#!/usr/bin/env python3
"""
Hush Engine Feedback Analyzer

Ingests user feedback from training/feedback/ in the repo.
Analyzes detection discrepancies and generates Claude-actionable recommendations.

Output: JSON file with structured action items for engine improvement.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional

# Add hush_engine to path
HUSH_ENGINE_PATH = Path(__file__).parent.parent / "hush_engine"
sys.path.insert(0, str(HUSH_ENGINE_PATH))

from detectors.pii_detector import PIIDetector

# Paths - use repo-relative paths
REPO_ROOT = Path(__file__).parent.parent
FEEDBACK_DIR = REPO_ROOT / "training" / "feedback"
OUTPUT_DIR = REPO_ROOT / "training" / "analysis"


class FeedbackAnalyzer:
    """Analyzes user feedback to identify detection improvements."""

    def __init__(self):
        self.detector = PIIDetector()
        self.feedback_items: List[Dict] = []
        self.analysis_results: Dict[str, Any] = {}

    def load_feedback(self) -> int:
        """Load all feedback files from the feedback directory."""
        if not FEEDBACK_DIR.exists():
            print(f"Feedback directory not found: {FEEDBACK_DIR}")
            return 0

        feedback_files = list(FEEDBACK_DIR.glob("*.json"))
        print(f"Found {len(feedback_files)} feedback files")

        for fb_file in sorted(feedback_files):
            try:
                with open(fb_file, 'r') as f:
                    data = json.load(f)
                    data['_feedback_file'] = fb_file.name
                    self.feedback_items.append(data)
            except Exception as e:
                print(f"Error loading {fb_file.name}: {e}")

        return len(self.feedback_items)

    def categorize_feedback(self) -> Dict[str, List[Dict]]:
        """Categorize feedback into actionable categories."""
        categories = {
            'false_positives': [],      # Detected as X, should be nothing
            'missed_detections': [],     # Not detected (CUSTOM), should be X
            'misclassifications': [],    # Detected as X, should be Y
            'confirmed_correct': [],     # User confirmed detection is correct
            'duplicates': [],            # Duplicate feedback entries
        }

        seen_items = set()

        for item in self.feedback_items:
            detected_type = item.get('detectedEntityType', '')
            suggested_types = item.get('suggestedEntityTypes', [])
            detected_text = item.get('detectedText', '')

            # Create dedup key
            dedup_key = (detected_text, detected_type, tuple(sorted(suggested_types)))

            if dedup_key in seen_items:
                categories['duplicates'].append(item)
                continue
            seen_items.add(dedup_key)

            # Categorize
            if detected_type == 'CUSTOM':
                # Engine didn't detect it, user says it should be something
                if suggested_types and suggested_types != ['CUSTOM']:
                    categories['missed_detections'].append(item)
            elif not suggested_types or suggested_types == []:
                # User says it shouldn't be detected at all
                categories['false_positives'].append(item)
            elif suggested_types == ['CUSTOM']:
                # User says it's custom/shouldn't be this type
                categories['false_positives'].append(item)
            elif detected_type in suggested_types:
                # User confirmed the detection is correct
                categories['confirmed_correct'].append(item)
            else:
                # Detected as one type, should be another
                categories['misclassifications'].append(item)

        return categories

    def analyze_patterns(self, categories: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze patterns in feedback to identify systematic issues."""
        patterns = {
            'false_positive_patterns': defaultdict(list),
            'missed_detection_patterns': defaultdict(list),
            'misclassification_patterns': defaultdict(list),
        }

        # Analyze false positives - what's being wrongly detected?
        for item in categories['false_positives']:
            detected_type = item.get('detectedEntityType', '')
            detected_text = item.get('detectedText', '')
            patterns['false_positive_patterns'][detected_type].append({
                'text': detected_text,
                'notes': item.get('notes', ''),
                'file': item.get('fileName', ''),
            })

        # Analyze missed detections - what should have been detected?
        for item in categories['missed_detections']:
            suggested = item.get('suggestedEntityTypes', [])
            for stype in suggested:
                patterns['missed_detection_patterns'][stype].append({
                    'text': item.get('detectedText', ''),
                    'notes': item.get('notes', ''),
                    'file': item.get('fileName', ''),
                })

        # Analyze misclassifications
        for item in categories['misclassifications']:
            detected_type = item.get('detectedEntityType', '')
            suggested = item.get('suggestedEntityTypes', [])
            key = f"{detected_type} -> {'/'.join(suggested)}"
            patterns['misclassification_patterns'][key].append({
                'text': item.get('detectedText', ''),
                'notes': item.get('notes', ''),
                'file': item.get('fileName', ''),
            })

        return patterns

    def re_analyze_texts(self, categories: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Re-run detection on feedback texts to verify current behavior."""
        verification = {
            'still_false_positives': [],
            'now_detected': [],
            'still_missed': [],
        }

        # Check if false positives are still being detected
        for item in categories['false_positives']:
            text = item.get('detectedText', '')
            if text:
                entities = self.detector.analyze_text(text)
                detected_types = [e.entity_type for e in entities]
                if item.get('detectedEntityType') in detected_types:
                    verification['still_false_positives'].append({
                        'text': text,
                        'detected_as': item.get('detectedEntityType'),
                        'current_detections': detected_types,
                    })

        # Check if missed detections are now caught
        for item in categories['missed_detections']:
            text = item.get('detectedText', '')
            suggested = item.get('suggestedEntityTypes', [])
            if text:
                entities = self.detector.analyze_text(text)
                detected_types = [e.entity_type for e in entities]
                if any(s in detected_types for s in suggested):
                    verification['now_detected'].append({
                        'text': text,
                        'expected': suggested,
                        'current_detections': detected_types,
                    })
                else:
                    verification['still_missed'].append({
                        'text': text,
                        'expected': suggested,
                        'current_detections': detected_types,
                    })

        return verification

    def generate_action_items(self, categories: Dict, patterns: Dict, verification: Dict) -> List[Dict]:
        """Generate Claude-actionable recommendations."""
        action_items = []

        # Action items for false positives
        fp_by_type = patterns['false_positive_patterns']
        for entity_type, examples in fp_by_type.items():
            if len(examples) >= 2:  # Only if pattern appears multiple times
                texts = [e['text'] for e in examples]
                action_items.append({
                    'priority': 'HIGH' if len(examples) >= 3 else 'MEDIUM',
                    'category': 'FALSE_POSITIVE',
                    'entity_type': entity_type,
                    'action': f"Add filter to exclude these patterns from {entity_type} detection",
                    'examples': texts[:5],
                    'count': len(examples),
                    'implementation_hint': f"Update _filter_false_positives() in pii_detector.py to filter {entity_type} patterns like: {', '.join(texts[:3])}",
                })

        # Action items for missed detections
        md_by_type = patterns['missed_detection_patterns']
        for entity_type, examples in md_by_type.items():
            if examples:
                texts = [e['text'] for e in examples]
                action_items.append({
                    'priority': 'HIGH' if len(examples) >= 2 else 'MEDIUM',
                    'category': 'MISSED_DETECTION',
                    'entity_type': entity_type,
                    'action': f"Improve {entity_type} detection to catch these patterns",
                    'examples': texts[:5],
                    'count': len(examples),
                    'implementation_hint': f"Add pattern recognizer or update existing {entity_type} patterns in pii_detector.py",
                })

        # Action items for misclassifications
        mc_patterns = patterns['misclassification_patterns']
        for key, examples in mc_patterns.items():
            if examples:
                texts = [e['text'] for e in examples]
                action_items.append({
                    'priority': 'MEDIUM',
                    'category': 'MISCLASSIFICATION',
                    'classification_error': key,
                    'action': f"Fix classification: {key}",
                    'examples': texts[:5],
                    'count': len(examples),
                    'implementation_hint': f"Review pattern priority or add disambiguation logic",
                })

        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        action_items.sort(key=lambda x: (priority_order.get(x['priority'], 99), -x['count']))

        return action_items

    def run_analysis(self) -> Dict[str, Any]:
        """Run full analysis and return results."""
        print("\n" + "="*70)
        print("HUSH ENGINE FEEDBACK ANALYSIS")
        print("="*70 + "\n")

        # Load feedback
        count = self.load_feedback()
        if count == 0:
            return {'error': 'No feedback found'}

        print(f"Loaded {count} feedback items\n")

        # Categorize
        categories = self.categorize_feedback()
        print("Categorization:")
        print(f"  - False positives: {len(categories['false_positives'])}")
        print(f"  - Missed detections: {len(categories['missed_detections'])}")
        print(f"  - Misclassifications: {len(categories['misclassifications'])}")
        print(f"  - Confirmed correct: {len(categories['confirmed_correct'])}")
        print(f"  - Duplicates: {len(categories['duplicates'])}")

        # Analyze patterns
        patterns = self.analyze_patterns(categories)

        # Re-analyze with current detector
        print("\nVerifying with current detector...")
        verification = self.re_analyze_texts(categories)
        print(f"  - Still false positives: {len(verification['still_false_positives'])}")
        print(f"  - Now detected: {len(verification['now_detected'])}")
        print(f"  - Still missed: {len(verification['still_missed'])}")

        # Generate action items
        action_items = self.generate_action_items(categories, patterns, verification)

        # Compile results
        self.analysis_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'detector_version': PIIDetector.getVersion(),
            'feedback_count': count,
            'summary': {
                'false_positives': len(categories['false_positives']),
                'missed_detections': len(categories['missed_detections']),
                'misclassifications': len(categories['misclassifications']),
                'confirmed_correct': len(categories['confirmed_correct']),
                'duplicates': len(categories['duplicates']),
            },
            'verification': {
                'still_false_positives': len(verification['still_false_positives']),
                'now_detected': len(verification['now_detected']),
                'still_missed': len(verification['still_missed']),
            },
            'action_items': action_items,
            'detailed_patterns': {
                'false_positive_patterns': {k: v for k, v in patterns['false_positive_patterns'].items()},
                'missed_detection_patterns': {k: v for k, v in patterns['missed_detection_patterns'].items()},
            },
            'raw_categories': {
                'false_positives': categories['false_positives'],
                'missed_detections': categories['missed_detections'],
                'misclassifications': categories['misclassifications'],
            }
        }

        return self.analysis_results

    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Save analysis results to JSON file."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = OUTPUT_DIR / f"analysis_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def print_action_items(self):
        """Print action items in a readable format."""
        action_items = self.analysis_results.get('action_items', [])

        if not action_items:
            print("\nNo action items identified.")
            return

        print("\n" + "="*70)
        print("ACTION ITEMS FOR CLAUDE")
        print("="*70)

        for i, item in enumerate(action_items, 1):
            priority = item['priority']
            marker = "🔴" if priority == 'HIGH' else "🟡" if priority == 'MEDIUM' else "🟢"

            print(f"\n{marker} [{priority}] Action Item #{i}")
            print(f"   Category: {item['category']}")
            print(f"   Action: {item['action']}")
            print(f"   Examples ({item['count']} total): {item['examples'][:3]}")
            print(f"   Hint: {item['implementation_hint']}")


def get_processed_feedback() -> set:
    """Get set of feedback files that have been processed (acted upon)."""
    processed_file = OUTPUT_DIR / "processed_feedback.json"
    if processed_file.exists():
        with open(processed_file, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_files', []))
    return set()


def mark_feedback_processed(feedback_files: List[str]):
    """Mark feedback files as processed after actions are taken."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_file = OUTPUT_DIR / "processed_feedback.json"

    existing = get_processed_feedback()
    existing.update(feedback_files)

    with open(processed_file, 'w') as f:
        json.dump({
            'processed_files': list(existing),
            'last_updated': datetime.now().isoformat()
        }, f, indent=2)

    print(f"Marked {len(feedback_files)} feedback files as processed")


def clear_feedback(only_processed: bool = True):
    """Clear feedback files (archive them).

    Args:
        only_processed: If True, only clear files that have been marked as processed.
                       If False, clear all feedback files.
    """
    if not FEEDBACK_DIR.exists():
        print("No feedback directory found.")
        return

    archive_dir = FEEDBACK_DIR / "archived"
    archive_dir.mkdir(exist_ok=True)

    feedback_files = list(FEEDBACK_DIR.glob("*.json"))
    if not feedback_files:
        print("No feedback files to clear.")
        return

    processed = get_processed_feedback()

    if only_processed:
        # Only archive files that have been processed
        files_to_archive = [f for f in feedback_files if f.name in processed]
        if not files_to_archive:
            print("No processed feedback files to clear.")
            print(f"  ({len(feedback_files)} unprocessed feedback files remain)")
            return
    else:
        files_to_archive = feedback_files

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = archive_dir / f"batch_{timestamp}"
    batch_dir.mkdir(exist_ok=True)

    for fb_file in files_to_archive:
        dest = batch_dir / fb_file.name
        fb_file.rename(dest)

    remaining = len(feedback_files) - len(files_to_archive)
    print(f"Archived {len(files_to_archive)} feedback files to: {batch_dir}")
    if remaining > 0:
        print(f"  ({remaining} unprocessed feedback files remain)")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Hush feedback and generate action items')
    parser.add_argument('--clear', action='store_true',
                       help='Clear ONLY processed feedback after analysis (archives them)')
    parser.add_argument('--clear-all', action='store_true',
                       help='Clear ALL feedback (use after acting on all items)')
    parser.add_argument('--clear-only', action='store_true',
                       help='Only clear feedback without analysis')
    parser.add_argument('--mark-processed', action='store_true',
                       help='Mark current feedback as processed (after implementing fixes)')
    args = parser.parse_args()

    if args.clear_only:
        clear_feedback(only_processed=not args.clear_all)
        return

    if args.mark_processed:
        # Mark all current feedback files as processed
        feedback_files = [f.name for f in FEEDBACK_DIR.glob("*.json")]
        if feedback_files:
            mark_feedback_processed(feedback_files)
        else:
            print("No feedback files to mark as processed.")
        return

    analyzer = FeedbackAnalyzer()
    results = analyzer.run_analysis()

    if 'error' not in results:
        analyzer.print_action_items()
        output_path = analyzer.save_results()

        # Also save a Claude-specific action file
        claude_actions = {
            'purpose': 'Hush Engine improvement actions based on user feedback',
            'instructions': 'Use these action items to improve pii_detector.py',
            'detector_file': 'hush_engine/detectors/pii_detector.py',
            'action_items': results['action_items'],
            'feedback_files': [item.get('_feedback_file') for item in analyzer.feedback_items],
        }

        claude_path = OUTPUT_DIR / "claude_actions.json"
        with open(claude_path, 'w') as f:
            json.dump(claude_actions, f, indent=2)
        print(f"Claude actions saved to: {claude_path}")

        # Clear feedback if requested
        if args.clear or args.clear_all:
            print("\n" + "-"*70)
            clear_feedback(only_processed=not args.clear_all)


if __name__ == "__main__":
    main()