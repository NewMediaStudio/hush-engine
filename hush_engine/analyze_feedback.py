#!/usr/bin/env python3
"""
Feedback Analyzer - Analyzes training feedback to improve PII detection
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime

# Import config for auto-adjustment
sys.path.insert(0, str(Path(__file__).parent))
from detection_config import get_config, DetectionConfig


class FeedbackAnalyzer:
    """Analyzes feedback logs to identify detection patterns and suggest improvements"""

    def __init__(self, feedback_path: str = None):
        self.feedback_path = feedback_path or self._find_feedback_file()
        self.entries: List[Dict] = []
        self.load_feedback()

    def _find_feedback_file(self) -> str:
        """Find the most recent feedback file (prefer ~/.hush)"""
        candidates = [
            Path.home() / ".hush" / "training_feedback.jsonl",
            Path.cwd() / "training_feedback.jsonl",
            Path.cwd() / "samples" / "training_feedback.jsonl",
            Path.home() / "Desktop" / "training_feedback.jsonl",
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        return str(candidates[0])  # Default

    def load_feedback(self):
        """Load all feedback entries from JSONL file"""
        self.entries = []
        path = Path(self.feedback_path)

        if not path.exists():
            return

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def analyze(self) -> Dict[str, Any]:
        """Run full analysis on feedback data"""
        if not self.entries:
            return {
                "status": "no_data",
                "message": "No feedback data found",
                "suggestions": []
            }

        analysis = {
            "status": "success",
            "total_sessions": len(self.entries),
            "total_edits": sum(e.get("edit_summary", {}).get("total_edits", 0) for e in self.entries),
            "false_positives": self._analyze_false_positives(),
            "missed_detections": self._analyze_missed_detections(),
            "entity_accuracy": self._analyze_entity_accuracy(),
            "suggestions": [],
            "timestamp": datetime.now().isoformat()
        }

        # Generate suggestions based on analysis
        analysis["suggestions"] = self._generate_suggestions(analysis)

        return analysis

    def _analyze_false_positives(self) -> Dict[str, Any]:
        """Analyze patterns in false positives (bars users removed)"""
        false_positives = []
        by_entity_type = defaultdict(list)

        for entry in self.entries:
            removed = entry.get("user_edits", {}).get("removed_bars", [])
            for item in removed:
                false_positives.append(item)
                entity_type = item.get("entity_type", "UNKNOWN")
                by_entity_type[entity_type].append(item.get("text", ""))

        return {
            "total": len(false_positives),
            "by_entity_type": {k: len(v) for k, v in by_entity_type.items()},
            "examples": {k: v[:5] for k, v in by_entity_type.items()}  # Top 5 examples per type
        }

    def _analyze_missed_detections(self) -> Dict[str, Any]:
        """Analyze patterns in missed detections (areas users added)"""
        missed = []
        bbox_sizes = []

        for entry in self.entries:
            added = entry.get("user_edits", {}).get("added_areas", [])
            for item in added:
                missed.append(item)
                bbox = item.get("bbox", [0, 0, 0, 0])
                if len(bbox) == 4:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    bbox_sizes.append({"width": width, "height": height, "area": width * height})

        avg_size = None
        if bbox_sizes:
            avg_size = {
                "avg_width": sum(b["width"] for b in bbox_sizes) / len(bbox_sizes),
                "avg_height": sum(b["height"] for b in bbox_sizes) / len(bbox_sizes),
                "avg_area": sum(b["area"] for b in bbox_sizes) / len(bbox_sizes)
            }

        return {
            "total": len(missed),
            "avg_bbox_size": avg_size,
            "count_by_session": [
                len(e.get("user_edits", {}).get("added_areas", []))
                for e in self.entries
            ]
        }

    def _analyze_entity_accuracy(self) -> Dict[str, Any]:
        """Calculate accuracy metrics per entity type"""
        entity_stats = defaultdict(lambda: {"detected": 0, "kept": 0, "removed": 0})

        for entry in self.entries:
            detections = entry.get("original_detections", [])
            removed_indices = {
                item.get("detection_index")
                for item in entry.get("user_edits", {}).get("removed_bars", [])
            }

            for det in detections:
                if det.get("was_selected"):
                    entity_type = det.get("entity_type", "UNKNOWN")
                    entity_stats[entity_type]["detected"] += 1

                    if det.get("index") in removed_indices:
                        entity_stats[entity_type]["removed"] += 1
                    else:
                        entity_stats[entity_type]["kept"] += 1

        # Calculate precision per entity type
        accuracy = {}
        for entity_type, stats in entity_stats.items():
            total = stats["detected"]
            if total > 0:
                precision = stats["kept"] / total
                accuracy[entity_type] = {
                    "total_detected": total,
                    "kept": stats["kept"],
                    "removed_as_false_positive": stats["removed"],
                    "precision": round(precision, 3)
                }

        return accuracy

    def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """Generate actionable suggestions based on analysis"""
        suggestions = []

        # Check for high false positive entity types
        fp = analysis.get("false_positives", {})
        by_type = fp.get("by_entity_type", {})

        for entity_type, count in by_type.items():
            if count >= 3:
                suggestions.append(
                    f"Consider raising confidence threshold for {entity_type} "
                    f"({count} false positives recorded)"
                )

        # Check entity accuracy
        accuracy = analysis.get("entity_accuracy", {})
        for entity_type, stats in accuracy.items():
            precision = stats.get("precision", 1.0)
            if precision < 0.7 and stats.get("total_detected", 0) >= 5:
                suggestions.append(
                    f"{entity_type} has low precision ({precision:.0%}). "
                    f"Review detection patterns or increase confidence threshold."
                )

        # Check for many missed detections
        missed = analysis.get("missed_detections", {})
        if missed.get("total", 0) >= 5:
            suggestions.append(
                f"{missed['total']} areas were manually added. "
                f"Consider adding custom patterns for commonly missed content."
            )

        # If no issues found
        if not suggestions:
            if analysis.get("total_edits", 0) == 0:
                suggestions.append("No edits recorded yet. Detection appears to be working well.")
            else:
                suggestions.append("Detection accuracy looks good based on current feedback.")

        return suggestions

    def auto_adjust_thresholds(self, min_samples: int = 3) -> Dict[str, Any]:
        """
        Auto-adjust detection thresholds based on false positive rates

        Args:
            min_samples: Minimum false positives needed to trigger adjustment

        Returns:
            Dict with adjustment results
        """
        analysis = self.analyze()

        if analysis["status"] != "success":
            return {"status": "no_data", "adjustments": []}

        # Calculate false positive rates per entity type
        accuracy = analysis.get("entity_accuracy", {})
        fp_rates = {}

        for entity_type, stats in accuracy.items():
            total = stats.get("total_detected", 0)
            removed = stats.get("removed_as_false_positive", 0)

            if total >= min_samples:
                fp_rates[entity_type] = removed / total if total > 0 else 0

        if not fp_rates:
            return {"status": "insufficient_data", "adjustments": [], "message": "Not enough samples yet"}

        # Apply adjustments via config
        config = get_config()
        adjustments = config.adjust_from_feedback(fp_rates, min_samples=min_samples)

        return {
            "status": "success",
            "adjustments": [
                {
                    "entity_type": et,
                    "old_threshold": old,
                    "new_threshold": new,
                    "false_positive_rate": fp
                }
                for et, old, new, fp in adjustments
            ],
            "message": f"Adjusted {len(adjustments)} thresholds" if adjustments else "No adjustments needed"
        }

    def print_report(self, auto_adjust: bool = True):
        """Print a formatted analysis report

        Args:
            auto_adjust: Whether to auto-adjust thresholds based on feedback
        """
        analysis = self.analyze()

        print("\n" + "=" * 60)
        print("FEEDBACK ANALYSIS REPORT")
        print("=" * 60)

        if analysis["status"] == "no_data":
            print(f"\n{analysis['message']}")
            return analysis

        print(f"\nSessions analyzed: {analysis['total_sessions']}")
        print(f"Total user edits: {analysis['total_edits']}")

        # False positives
        fp = analysis["false_positives"]
        print(f"\n--- FALSE POSITIVES (bars removed by user) ---")
        print(f"Total: {fp['total']}")
        if fp["by_entity_type"]:
            print("By entity type:")
            for entity_type, count in sorted(fp["by_entity_type"].items(), key=lambda x: -x[1]):
                examples = fp["examples"].get(entity_type, [])
                example_str = f' (e.g., "{examples[0]}")' if examples else ""
                print(f"  - {entity_type}: {count}{example_str}")

        # Missed detections
        missed = analysis["missed_detections"]
        print(f"\n--- MISSED DETECTIONS (areas added by user) ---")
        print(f"Total: {missed['total']}")
        if missed["avg_bbox_size"]:
            size = missed["avg_bbox_size"]
            print(f"Average size: {size['avg_width']:.0f}x{size['avg_height']:.0f} pixels")

        # Entity accuracy
        accuracy = analysis["entity_accuracy"]
        if accuracy:
            print(f"\n--- ENTITY TYPE ACCURACY ---")
            for entity_type, stats in sorted(accuracy.items(), key=lambda x: x[1]["precision"]):
                print(f"  {entity_type}: {stats['precision']:.0%} precision "
                      f"({stats['kept']}/{stats['total_detected']} kept)")

        # Suggestions
        print(f"\n--- SUGGESTIONS ---")
        for i, suggestion in enumerate(analysis["suggestions"], 1):
            print(f"{i}. {suggestion}")

        # Auto-adjust thresholds if enabled
        if auto_adjust:
            print(f"\n--- AUTO-ADJUSTMENT ---")
            adj_result = self.auto_adjust_thresholds()
            if adj_result["status"] == "success" and adj_result["adjustments"]:
                print("Thresholds adjusted based on feedback:")
                for adj in adj_result["adjustments"]:
                    print(f"  - {adj['entity_type']}: {adj['old_threshold']:.2f} -> {adj['new_threshold']:.2f} "
                          f"(FP rate: {adj['false_positive_rate']:.0%})")
            else:
                print(adj_result.get("message", "No adjustments made"))

        print("\n" + "=" * 60)

        return analysis

    def save_report(self, output_path: str = None):
        """Save analysis report to JSON file"""
        analysis = self.analyze()

        if output_path is None:
            output_path = str(Path(self.feedback_path).parent / "feedback_analysis.json")

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"Report saved to: {output_path}")
        return output_path


def analyze_and_report(feedback_path: str = None) -> Dict[str, Any]:
    """Convenience function to run analysis and print report"""
    analyzer = FeedbackAnalyzer(feedback_path)
    return analyzer.print_report()


def main():
    """CLI entry point"""
    feedback_path = sys.argv[1] if len(sys.argv) > 1 else None

    analyzer = FeedbackAnalyzer(feedback_path)
    analysis = analyzer.print_report()

    # Optionally save report
    if "--save" in sys.argv:
        analyzer.save_report()

    return 0 if analysis["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
