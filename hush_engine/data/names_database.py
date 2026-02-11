#!/usr/bin/env python3
"""
Names Database for Hush Engine

Unified name lookup from curated names + popular names across 100+ countries.
All name data lives in the generated file (popular_names/generated_popular_names.py),
which merges curated names (curated_names.py) with the popular-names-by-country dataset.

To regenerate after editing curated names:
    python3 tools/ingest_popular_names.py --local

Usage:
    from hush_engine.data.names_database import NamesDatabase
    db = NamesDatabase()
    db.is_first_name("John")  # True
    db.is_last_name("Smith")  # True
    db.is_name("Maria")       # True
"""

from typing import Set, Dict, Any
import re

from hush_engine.data.popular_names.generated_popular_names import (
    ALL_FIRST_NAMES,
    ALL_LAST_NAMES,
    FORENAMES_BY_LOCALE,
    SURNAMES_BY_LOCALE,
)

# Name titles (prefixes that indicate a name follows)
NAME_TITLES = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "lord", "lady",
    "monsieur", "madame", "mademoiselle",
    "herr", "frau", "fraulein",
    "senor", "senora", "senorita", "don", "dona",
    "signor", "signora", "signorina",
    "senhor", "senhora",
    "rabbi", "father", "sister", "brother", "reverend", "pastor", "imam", "sheikh",
    "captain", "colonel", "general", "sergeant", "lieutenant", "major", "admiral",
}


class NamesDatabase:
    """
    Comprehensive names database for fast name lookup.

    Loads curated + popular names from 100+ countries organized by locale.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.first_names: Set[str] = ALL_FIRST_NAMES
        self.last_names: Set[str] = ALL_LAST_NAMES
        self.all_names: Set[str] = self.first_names | self.last_names
        self.forenames_by_locale: Dict[str, Set[str]] = FORENAMES_BY_LOCALE
        self.surnames_by_locale: Dict[str, Set[str]] = SURNAMES_BY_LOCALE
        self.titles: Set[str] = NAME_TITLES

    def is_first_name(self, name: str) -> bool:
        return name.lower().strip() in self.first_names

    def is_last_name(self, name: str) -> bool:
        return name.lower().strip() in self.last_names

    def is_name(self, name: str) -> bool:
        return name.lower().strip() in self.all_names

    def is_title(self, word: str) -> bool:
        return word.lower().strip().rstrip('.') in self.titles

    def check_name(self, name: str) -> Dict[str, Any]:
        """Check a name and return detailed info with confidence score."""
        name_lower = name.lower().strip()
        is_first = name_lower in self.first_names
        is_last = name_lower in self.last_names

        if is_first and is_last:
            confidence = 0.95
        elif is_first or is_last:
            confidence = 0.85
        else:
            confidence = 0.0

        return {
            "name": name,
            "is_first_name": is_first,
            "is_last_name": is_last,
            "is_name": is_first or is_last,
            "confidence": confidence,
        }

    def find_names_in_text(self, text: str) -> list:
        """Find potential names in text. Returns (name, start, end, confidence) tuples."""
        results = []
        for match in re.finditer(r'\b([A-Z][a-z]+)\b', text):
            word = match.group(1)
            info = self.check_name(word)
            if info["is_name"]:
                results.append((word, match.start(), match.end(), info["confidence"]))
        return results

    def stats(self) -> Dict[str, int]:
        return {
            "total_first_names": len(self.first_names),
            "total_last_names": len(self.last_names),
            "total_unique_names": len(self.all_names),
            "total_titles": len(self.titles),
            "forename_locales": len(self.forenames_by_locale),
            "surname_locales": len(self.surnames_by_locale),
        }


def get_names_database() -> NamesDatabase:
    return NamesDatabase()
