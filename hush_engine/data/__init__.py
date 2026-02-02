"""
Data modules for Hush Engine.

Provides lightweight databases and resources for PII detection.
"""

from .names_database import NamesDatabase, get_names_database

__all__ = [
    "NamesDatabase",
    "get_names_database",
]
