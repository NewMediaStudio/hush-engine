#!/usr/bin/env python3
"""
Library Manager for Hush Engine

Manages optional libraries (phonenumbers, spacy, etc.) with:
- Status tracking (installed, enabled, downloading)
- Enable/disable functionality
- Download with progress notifications
- Lazy loading for disabled libraries
- Config persistence in ~/.hush/config.json
"""

import json
import subprocess
import sys
import importlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum


class LibraryStatus(Enum):
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    DOWNLOADING = "downloading"
    FAILED = "failed"


@dataclass
class LibraryInfo:
    """Information about an optional library."""
    name: str
    pip_package: str
    description: str
    status: str = "not_installed"
    enabled: bool = True
    version: Optional[str] = None
    download_progress: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Registry of optional libraries
OPTIONAL_LIBRARIES = {
    "phonenumbers": LibraryInfo(
        name="phonenumbers",
        pip_package="phonenumbers",
        description="International phone number parsing and validation (Google's libphonenumber)",
    ),
    "spacy": LibraryInfo(
        name="spacy",
        pip_package="spacy",
        description="Industrial-strength NLP for named entity recognition",
    ),
    "spacy_en_core_web_sm": LibraryInfo(
        name="spacy_en_core_web_sm",
        pip_package="en_core_web_sm",
        description="SpaCy English language model (small)",
    ),
    "rapidfuzz": LibraryInfo(
        name="rapidfuzz",
        pip_package="rapidfuzz",
        description="Fast fuzzy string matching for company/name detection",
    ),
}


class LibraryManager:
    """Manages optional libraries for Hush Engine."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for library manager."""
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
        self.libraries: Dict[str, LibraryInfo] = {}
        self._download_threads: Dict[str, threading.Thread] = {}
        self._progress_callback: Optional[Callable[[str, float, str], None]] = None

        # Initialize library registry
        self._init_libraries()

        # Load saved config
        self._load_config()

        # Check which libraries are installed
        self._detect_installed_libraries()

    def _init_libraries(self):
        """Initialize library registry from template."""
        for name, template in OPTIONAL_LIBRARIES.items():
            self.libraries[name] = LibraryInfo(
                name=template.name,
                pip_package=template.pip_package,
                description=template.description,
            )

    def _load_config(self):
        """Load library configuration from disk."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            library_config = config.get("libraries", {})
            for name, settings in library_config.items():
                if name in self.libraries:
                    self.libraries[name].enabled = settings.get("enabled", True)
        except (json.JSONDecodeError, IOError) as e:
            sys.stderr.write(f"[LibraryManager] Failed to load config: {e}\n")

    def _save_config(self):
        """Save library configuration to disk."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new
        config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                config = {}

        # Update library settings
        config["libraries"] = {
            name: {"enabled": lib.enabled}
            for name, lib in self.libraries.items()
        }

        # Save
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _detect_installed_libraries(self):
        """Check which libraries are currently installed."""
        for name, lib in self.libraries.items():
            try:
                if name == "spacy_en_core_web_sm":
                    # Special handling for spacy models
                    import spacy
                    try:
                        spacy.load("en_core_web_sm")
                        lib.status = LibraryStatus.INSTALLED.value
                        lib.version = "installed"
                    except OSError:
                        lib.status = LibraryStatus.NOT_INSTALLED.value
                else:
                    module = importlib.import_module(name)
                    lib.status = LibraryStatus.INSTALLED.value
                    lib.version = getattr(module, '__version__', 'unknown')
            except ImportError:
                lib.status = LibraryStatus.NOT_INSTALLED.value

    def get_library_status(self, library_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of libraries.

        Args:
            library_name: Specific library name, or None for all libraries

        Returns:
            Dict with library status information
        """
        if library_name:
            if library_name not in self.libraries:
                return {"error": f"Unknown library: {library_name}"}
            return {"libraries": {library_name: self.libraries[library_name].to_dict()}}

        return {
            "libraries": {
                name: lib.to_dict() for name, lib in self.libraries.items()
            }
        }

    def set_library_enabled(self, library_name: str, enabled: bool) -> Dict[str, Any]:
        """
        Enable or disable a library.

        Args:
            library_name: Name of the library
            enabled: Whether to enable or disable

        Returns:
            Dict with result status
        """
        if library_name not in self.libraries:
            return {"success": False, "error": f"Unknown library: {library_name}"}

        lib = self.libraries[library_name]
        lib.enabled = enabled
        self._save_config()

        return {
            "success": True,
            "library": library_name,
            "enabled": enabled,
            "status": lib.status,
        }

    def is_library_available(self, library_name: str) -> bool:
        """
        Check if a library is both installed and enabled.

        Args:
            library_name: Name of the library

        Returns:
            True if library is installed and enabled
        """
        if library_name not in self.libraries:
            return False

        lib = self.libraries[library_name]
        return lib.status == LibraryStatus.INSTALLED.value and lib.enabled

    def download_library(
        self,
        library_name: str,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Download/install a library with pip.

        Args:
            library_name: Name of the library to install
            progress_callback: Callback(library_name, progress, message)

        Returns:
            Dict with download result
        """
        if library_name not in self.libraries:
            return {"success": False, "error": f"Unknown library: {library_name}"}

        lib = self.libraries[library_name]

        # Check if already downloading
        if lib.status == LibraryStatus.DOWNLOADING.value:
            return {"success": False, "error": "Download already in progress"}

        # Check if already installed
        if lib.status == LibraryStatus.INSTALLED.value:
            return {"success": True, "message": "Library already installed"}

        # Start download
        lib.status = LibraryStatus.DOWNLOADING.value
        lib.download_progress = 0.0
        lib.error_message = None

        def run_install():
            try:
                if progress_callback:
                    progress_callback(library_name, 0.1, "Starting installation...")

                lib.download_progress = 0.1

                # Use pip to install
                if library_name == "spacy_en_core_web_sm":
                    # SpaCy model needs special download command
                    cmd = [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
                else:
                    cmd = [sys.executable, "-m", "pip", "install", lib.pip_package]

                if progress_callback:
                    progress_callback(library_name, 0.3, f"Installing {lib.pip_package}...")

                lib.download_progress = 0.3

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode == 0:
                    lib.status = LibraryStatus.INSTALLED.value
                    lib.download_progress = 1.0

                    # Re-detect version
                    self._detect_installed_libraries()

                    if progress_callback:
                        progress_callback(library_name, 1.0, "Installation complete")
                else:
                    lib.status = LibraryStatus.FAILED.value
                    lib.error_message = result.stderr[:500] if result.stderr else "Installation failed"

                    if progress_callback:
                        progress_callback(library_name, -1, lib.error_message)

            except subprocess.TimeoutExpired:
                lib.status = LibraryStatus.FAILED.value
                lib.error_message = "Installation timed out"
                if progress_callback:
                    progress_callback(library_name, -1, "Installation timed out")
            except Exception as e:
                lib.status = LibraryStatus.FAILED.value
                lib.error_message = str(e)
                if progress_callback:
                    progress_callback(library_name, -1, str(e))
            finally:
                if library_name in self._download_threads:
                    del self._download_threads[library_name]

        # Run in background thread
        thread = threading.Thread(target=run_install, daemon=True)
        self._download_threads[library_name] = thread
        thread.start()

        return {
            "success": True,
            "message": "Download started",
            "library": library_name,
        }

    def download_library_sync(self, library_name: str) -> Dict[str, Any]:
        """
        Download/install a library synchronously (blocking).

        Args:
            library_name: Name of the library to install

        Returns:
            Dict with download result
        """
        if library_name not in self.libraries:
            return {"success": False, "error": f"Unknown library: {library_name}"}

        lib = self.libraries[library_name]

        # Check if already installed
        if lib.status == LibraryStatus.INSTALLED.value:
            return {"success": True, "message": "Library already installed"}

        try:
            lib.status = LibraryStatus.DOWNLOADING.value

            if library_name == "spacy_en_core_web_sm":
                cmd = [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", lib.pip_package]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                lib.status = LibraryStatus.INSTALLED.value
                self._detect_installed_libraries()
                return {"success": True, "message": "Installation complete"}
            else:
                lib.status = LibraryStatus.FAILED.value
                lib.error_message = result.stderr[:500] if result.stderr else "Installation failed"
                return {"success": False, "error": lib.error_message}

        except subprocess.TimeoutExpired:
            lib.status = LibraryStatus.FAILED.value
            lib.error_message = "Installation timed out"
            return {"success": False, "error": "Installation timed out"}
        except Exception as e:
            lib.status = LibraryStatus.FAILED.value
            lib.error_message = str(e)
            return {"success": False, "error": str(e)}


# Singleton instance
def get_library_manager() -> LibraryManager:
    """Get the singleton LibraryManager instance."""
    return LibraryManager()
