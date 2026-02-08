#!/usr/bin/env python3
"""
LightGBM Model Pre-loader

CRITICAL: This module MUST be imported BEFORE spaCy/presidio to avoid
OpenMP library conflicts (libomp vs libiomp5 on macOS).

The conflict occurs when:
1. spaCy loads its OpenMP runtime (libomp.dylib)
2. LightGBM then tries to load its OpenMP runtime (libiomp5.dylib)
3. Both runtimes conflict, causing segfaults

Solution: Import this module first to load LightGBM models before spaCy.

Usage in pii_detector.py:
    # At the VERY TOP of the file, before any presidio/spacy imports
    from hush_engine.detectors import lgbm_preloader  # noqa: F401
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(__file__).parent.parent / "models" / "lgbm"

# Pre-loaded models (global singletons)
_address_model = None
_person_model = None
_organization_model = None
_location_model = None

# Availability flags
PRELOAD_SUCCESSFUL = False


def _preload_all_models():
    """Pre-load all LightGBM models at import time."""
    global _address_model, _person_model, _organization_model, _location_model
    global PRELOAD_SUCCESSFUL

    try:
        import lightgbm as lgbm

        loaded_count = 0

        # Load ADDRESS model
        address_path = MODEL_DIR / "address_classifier.txt"
        if address_path.exists():
            try:
                _address_model = lgbm.Booster(model_file=str(address_path))
                loaded_count += 1
                logger.debug("Pre-loaded ADDRESS LightGBM model")
            except Exception as e:
                logger.warning(f"Failed to load ADDRESS model: {e}")

        # Load PERSON model
        person_path = MODEL_DIR / "person_classifier.txt"
        if person_path.exists():
            try:
                _person_model = lgbm.Booster(model_file=str(person_path))
                loaded_count += 1
                logger.debug("Pre-loaded PERSON LightGBM model")
            except Exception as e:
                logger.warning(f"Failed to load PERSON model: {e}")

        # Load ORGANIZATION model
        org_path = MODEL_DIR / "organization_classifier.txt"
        if org_path.exists():
            try:
                _organization_model = lgbm.Booster(model_file=str(org_path))
                loaded_count += 1
                logger.debug("Pre-loaded ORGANIZATION LightGBM model")
            except Exception as e:
                logger.warning(f"Failed to load ORGANIZATION model: {e}")

        # Load LOCATION model
        location_path = MODEL_DIR / "location_classifier.txt"
        if location_path.exists():
            try:
                _location_model = lgbm.Booster(model_file=str(location_path))
                loaded_count += 1
                logger.debug("Pre-loaded LOCATION LightGBM model")
            except Exception as e:
                logger.warning(f"Failed to load LOCATION model: {e}")

        if loaded_count > 0:
            PRELOAD_SUCCESSFUL = True
            print(f"Pre-loaded {loaded_count} LightGBM models (OpenMP-safe)", file=sys.stderr)
        else:
            logger.debug("No LightGBM models found to pre-load")

    except ImportError:
        logger.debug("LightGBM not installed, skipping pre-load")
    except Exception as e:
        logger.warning(f"LightGBM pre-load failed: {e}")


# Accessor functions
def get_address_model():
    """Get pre-loaded ADDRESS model."""
    return _address_model


def get_person_model():
    """Get pre-loaded PERSON model."""
    return _person_model


def get_organization_model():
    """Get pre-loaded ORGANIZATION model."""
    return _organization_model


def get_location_model():
    """Get pre-loaded LOCATION model."""
    return _location_model


def is_preload_successful():
    """Check if pre-loading was successful."""
    return PRELOAD_SUCCESSFUL


# Pre-load models immediately when this module is imported
_preload_all_models()
