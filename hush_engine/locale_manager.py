#!/usr/bin/env python3
"""
Locale Manager for Hush Engine

Manages locale settings for PII detection:
- Region-specific phone number parsing
- Address format detection
- Date format preferences
- Currency patterns
- Config persistence in ~/.hush/config.json
"""

import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


# Supported locales with their configurations
SUPPORTED_LOCALES = {
    "en_US": {
        "name": "English (United States)",
        "phone_region": "US",
        "date_formats": ["MM/DD/YYYY", "MM-DD-YYYY"],
        "currency": "USD",
        "address_format": "us",
    },
    "en_GB": {
        "name": "English (United Kingdom)",
        "phone_region": "GB",
        "date_formats": ["DD/MM/YYYY", "DD-MM-YYYY"],
        "currency": "GBP",
        "address_format": "uk",
    },
    "en_CA": {
        "name": "English (Canada)",
        "phone_region": "CA",
        "date_formats": ["YYYY-MM-DD", "DD/MM/YYYY"],
        "currency": "CAD",
        "address_format": "ca",
    },
    "en_AU": {
        "name": "English (Australia)",
        "phone_region": "AU",
        "date_formats": ["DD/MM/YYYY"],
        "currency": "AUD",
        "address_format": "au",
    },
    "de_DE": {
        "name": "German (Germany)",
        "phone_region": "DE",
        "date_formats": ["DD.MM.YYYY"],
        "currency": "EUR",
        "address_format": "de",
    },
    "fr_FR": {
        "name": "French (France)",
        "phone_region": "FR",
        "date_formats": ["DD/MM/YYYY"],
        "currency": "EUR",
        "address_format": "fr",
    },
    "es_ES": {
        "name": "Spanish (Spain)",
        "phone_region": "ES",
        "date_formats": ["DD/MM/YYYY"],
        "currency": "EUR",
        "address_format": "es",
    },
    "it_IT": {
        "name": "Italian (Italy)",
        "phone_region": "IT",
        "date_formats": ["DD/MM/YYYY"],
        "currency": "EUR",
        "address_format": "it",
    },
    "ja_JP": {
        "name": "Japanese (Japan)",
        "phone_region": "JP",
        "date_formats": ["YYYY/MM/DD"],
        "currency": "JPY",
        "address_format": "jp",
    },
    "zh_CN": {
        "name": "Chinese (China)",
        "phone_region": "CN",
        "date_formats": ["YYYY-MM-DD"],
        "currency": "CNY",
        "address_format": "cn",
    },
    "auto": {
        "name": "Auto-detect",
        "phone_region": None,  # Will try to detect from content
        "date_formats": None,
        "currency": None,
        "address_format": None,
    },
}


@dataclass
class LocaleSettings:
    """Current locale settings."""
    locale: str = "auto"
    phone_region: Optional[str] = None
    date_formats: Optional[List[str]] = None
    currency: Optional[str] = None
    address_format: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocaleManager:
    """Manages locale settings for Hush Engine."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for locale manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.config_path = Path.home() / ".hush" / "config.json"
        self.settings = LocaleSettings()

        # Load saved config
        self._load_config()

    def _load_config(self):
        """Load locale configuration from disk."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            locale_config = config.get("locale", {})
            locale = locale_config.get("locale", "auto")

            if locale in SUPPORTED_LOCALES:
                self._apply_locale(locale)
        except (json.JSONDecodeError, IOError):
            pass

    def _save_config(self):
        """Save locale configuration to disk."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new
        config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                config = {}

        # Update locale settings
        config["locale"] = {
            "locale": self.settings.locale,
        }

        # Save
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _apply_locale(self, locale: str):
        """Apply a locale configuration."""
        if locale not in SUPPORTED_LOCALES:
            raise ValueError(f"Unsupported locale: {locale}")

        locale_config = SUPPORTED_LOCALES[locale]
        self.settings.locale = locale
        self.settings.phone_region = locale_config.get("phone_region")
        self.settings.date_formats = locale_config.get("date_formats")
        self.settings.currency = locale_config.get("currency")
        self.settings.address_format = locale_config.get("address_format")

    def get_locale(self) -> Dict[str, Any]:
        """
        Get current locale settings.

        Returns:
            Dict with current locale info and available locales
        """
        return {
            "current": {
                "locale": self.settings.locale,
                "name": SUPPORTED_LOCALES.get(self.settings.locale, {}).get("name", "Unknown"),
                "phone_region": self.settings.phone_region,
                "date_formats": self.settings.date_formats,
                "currency": self.settings.currency,
                "address_format": self.settings.address_format,
            },
            "available": [
                {
                    "locale": key,
                    "name": val["name"],
                }
                for key, val in SUPPORTED_LOCALES.items()
            ]
        }

    def set_locale(self, locale: str) -> Dict[str, Any]:
        """
        Set locale preference.

        Args:
            locale: Locale code (e.g., "en_US", "de_DE", "auto")

        Returns:
            Dict with result status
        """
        if locale not in SUPPORTED_LOCALES:
            return {
                "success": False,
                "error": f"Unsupported locale: {locale}",
                "available": list(SUPPORTED_LOCALES.keys()),
            }

        self._apply_locale(locale)
        self._save_config()

        return {
            "success": True,
            "locale": locale,
            "name": SUPPORTED_LOCALES[locale]["name"],
            "settings": self.settings.to_dict(),
        }

    def get_phone_region(self) -> Optional[str]:
        """Get the phone region for number parsing."""
        return self.settings.phone_region

    def get_date_formats(self) -> Optional[List[str]]:
        """Get date formats for the current locale."""
        return self.settings.date_formats

    def get_currency(self) -> Optional[str]:
        """Get currency code for the current locale."""
        return self.settings.currency


# Singleton instance
def get_locale_manager() -> LocaleManager:
    """Get the singleton LocaleManager instance."""
    return LocaleManager()
