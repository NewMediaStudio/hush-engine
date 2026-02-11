#!/usr/bin/env python3
"""
Ingest and unify all name data for Hush Engine's NamesDatabase.

Merges two sources into a single locale-linked output:
  1. Curated names (currently inline in names_database.py)
  2. Popular names from sigpwned/popular-names-by-country-dataset (CC0)

Outputs: hush_engine/data/popular_names/generated_popular_names.py
  - FORENAMES_BY_LOCALE: dict[locale_code, set[str]]
  - SURNAMES_BY_LOCALE:  dict[locale_code, set[str]]
  - ALL_FIRST_NAMES:     flat set (all locales combined)
  - ALL_LAST_NAMES:      flat set (all locales combined)

Usage:
    python3 tools/ingest_popular_names.py --local        # use cached CSVs
    python3 tools/ingest_popular_names.py --stats-only   # just print stats
"""

import argparse
import csv
import io
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "hush_engine" / "data" / "popular_names"

FORENAMES_URL = "https://raw.githubusercontent.com/sigpwned/popular-names-by-country-dataset/master/common-forenames-by-country.csv"
SURNAMES_URL = "https://raw.githubusercontent.com/sigpwned/popular-names-by-country-dataset/master/common-surnames-by-country.csv"

FORENAMES_FILE = DATA_DIR / "common-forenames-by-country.csv"
SURNAMES_FILE = DATA_DIR / "common-surnames-by-country.csv"

# Map ISO 3166-1 alpha-2 country codes to locale/region groups
COUNTRY_TO_LOCALE = {
    # English-speaking
    "AU": "en", "CA": "en", "GB": "en", "IE": "en", "NZ": "en", "US": "en",
    "ZA": "en", "JM": "en", "TT": "en", "BB": "en", "GY": "en",
    # Spanish-speaking
    "AR": "es", "BO": "es", "CL": "es", "CO": "es", "CR": "es", "CU": "es",
    "DO": "es", "EC": "es", "SV": "es", "GT": "es", "HN": "es", "MX": "es",
    "NI": "es", "PA": "es", "PY": "es", "PE": "es", "PR": "es", "ES": "es",
    "UY": "es", "VE": "es", "GQ": "es",
    # Portuguese-speaking
    "BR": "pt", "PT": "pt", "AO": "pt", "MZ": "pt", "CV": "pt",
    # French-speaking
    "FR": "fr", "BE": "fr", "CH": "fr", "LU": "fr", "MC": "fr",
    "SN": "fr", "CI": "fr", "ML": "fr", "BF": "fr", "NE": "fr",
    "TD": "fr", "GA": "fr", "CG": "fr", "CD": "fr", "CM": "fr",
    "MG": "fr", "HT": "fr", "RE": "fr",
    # German-speaking
    "DE": "de", "AT": "de", "LI": "de",
    # Italian
    "IT": "it", "SM": "it", "VA": "it",
    # Dutch
    "NL": "nl", "SR": "nl",
    # Nordic
    "DK": "da", "FI": "fi", "IS": "is", "NO": "no", "SE": "sv",
    # Eastern European / Slavic
    "RU": "ru", "UA": "uk", "BY": "be", "PL": "pl", "CZ": "cs",
    "SK": "sk", "BG": "bg", "RS": "sr", "HR": "hr", "SI": "sl",
    "BA": "bs", "ME": "sr", "MK": "mk", "MD": "ro",
    # Baltic
    "LT": "lt", "LV": "lv", "EE": "et",
    # Romanian
    "RO": "ro",
    # Greek
    "GR": "el", "CY": "el",
    # Turkish
    "TR": "tr", "AZ": "az",
    # Arabic-speaking
    "SA": "ar", "AE": "ar", "EG": "ar", "IQ": "ar", "JO": "ar",
    "KW": "ar", "LB": "ar", "LY": "ar", "MA": "ar", "OM": "ar",
    "QA": "ar", "SY": "ar", "TN": "ar", "YE": "ar", "BH": "ar",
    "DZ": "ar", "SD": "ar", "PS": "ar",
    # Persian
    "IR": "fa", "AF": "fa",
    # South Asian
    "IN": "hi", "PK": "ur", "BD": "bn", "LK": "si", "NP": "ne",
    # Southeast Asian
    "TH": "th", "VN": "vi", "ID": "id", "MY": "ms", "PH": "tl",
    "MM": "my", "KH": "km", "LA": "lo", "SG": "ms",
    # East Asian
    "CN": "zh", "TW": "zh", "HK": "zh", "JP": "ja", "KR": "ko",
    "MN": "mn",
    # Central Asian
    "KZ": "kk", "UZ": "uz", "TJ": "tg", "KG": "ky", "TM": "tk",
    # Caucasus
    "GE": "ka", "AM": "hy",
    # Sub-Saharan Africa
    "NG": "yo", "GH": "ak", "KE": "sw", "TZ": "sw", "UG": "sw",
    "ET": "am", "RW": "rw", "ZW": "sn", "BW": "tn", "NA": "af",
    "MW": "ny", "ZM": "ny", "LS": "st", "SZ": "ss",
    # Caribbean / Pacific
    "FJ": "fj", "PG": "tpi", "WS": "sm",
    # Micro-states / other
    "AD": "ca", "MT": "mt", "HU": "hu", "AL": "sq",
    "XK": "sq",
}

# Locale display labels
LOCALE_NAMES = {
    "en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French",
    "de": "German", "it": "Italian", "nl": "Dutch", "da": "Danish",
    "fi": "Finnish", "is": "Icelandic", "no": "Norwegian", "sv": "Swedish",
    "ru": "Russian", "uk": "Ukrainian", "be": "Belarusian", "pl": "Polish",
    "cs": "Czech", "sk": "Slovak", "bg": "Bulgarian", "sr": "Serbian",
    "hr": "Croatian", "sl": "Slovenian", "bs": "Bosnian", "mk": "Macedonian",
    "ro": "Romanian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
    "el": "Greek", "tr": "Turkish", "az": "Azerbaijani", "ar": "Arabic",
    "fa": "Persian", "hi": "Hindi/Indic", "ur": "Urdu", "bn": "Bengali",
    "si": "Sinhala", "ne": "Nepali", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian", "ms": "Malay", "tl": "Filipino", "my": "Burmese",
    "km": "Khmer", "lo": "Lao", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "mn": "Mongolian", "kk": "Kazakh", "uz": "Uzbek",
    "tg": "Tajik", "ky": "Kyrgyz", "tk": "Turkmen", "ka": "Georgian",
    "hy": "Armenian", "yo": "Yoruba", "ak": "Akan", "sw": "Swahili",
    "am": "Amharic", "rw": "Kinyarwanda", "sn": "Shona", "tn": "Tswana",
    "af": "Afrikaans", "ny": "Chichewa", "st": "Sesotho", "ss": "Swazi",
    "fj": "Fijian", "tpi": "Tok Pisin", "sm": "Samoan", "ca": "Catalan",
    "mt": "Maltese", "hu": "Hungarian", "sq": "Albanian",
}


def is_latin(text: str) -> bool:
    """Check if text is primarily Latin script (allowing accented characters)."""
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "LATIN" not in name and name != "":
                return False
    return True


def strip_accents(text: str) -> str:
    """Strip Unicode accents."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_name(name: str) -> str:
    """Normalize a name: lowercase, strip accents, remove non-alpha chars."""
    name = name.strip()
    if not name:
        return ""
    name = strip_accents(name)
    name = name.lower()
    name = re.sub(r"[^a-z \-']", "", name)
    return name.strip()


def download_csv(url: str, dest: Path) -> str:
    """Download a CSV file using curl."""
    print(f"  Downloading {url}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-sL", url, "-o", str(dest)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  ERROR: curl failed: {result.stderr}")
        sys.exit(1)
    content = dest.read_text(encoding="utf-8-sig")
    print(f"  Saved to {dest} ({len(content):,} bytes)")
    return content


def load_csv(path: Path) -> str:
    """Load CSV from local file."""
    return path.read_text(encoding="utf-8-sig")


def parse_forenames(content: str) -> dict:
    """Parse forenames CSV. Returns {country_code: set of romanized names}."""
    reader = csv.DictReader(io.StringIO(content))
    by_country = {}
    for row in reader:
        country = row.get("Country", "").strip()
        romanized = row.get("Romanized Name", "").strip()
        if not country or not romanized:
            continue
        if not is_latin(romanized):
            continue
        name = normalize_name(romanized)
        if not name or len(name) < 2:
            continue
        by_country.setdefault(country, set()).add(name)
    return by_country


def parse_surnames(content: str) -> dict:
    """Parse surnames CSV. Returns {country_code: set of romanized names}."""
    reader = csv.DictReader(io.StringIO(content))
    by_country = {}
    for row in reader:
        country = row.get("Country", "").strip()
        romanized = row.get("Romanized Name", "").strip()
        if not country or not romanized:
            continue
        if not is_latin(romanized):
            continue
        name = normalize_name(romanized)
        if not name or len(name) < 2:
            continue
        by_country.setdefault(country, set()).add(name)
    return by_country


def get_curated_names() -> tuple:
    """Import curated names from curated_names.py.

    Returns (forenames_by_locale, surnames_by_locale) dicts.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from hush_engine.data.curated_names import CURATED_FIRST_NAMES, CURATED_LAST_NAMES
    return CURATED_FIRST_NAMES, CURATED_LAST_NAMES


def group_by_locale(by_country: dict) -> tuple:
    """Group country-level names into locale-level sets."""
    by_locale = {}
    unmapped = set()
    for country, names in by_country.items():
        locale = COUNTRY_TO_LOCALE.get(country)
        if locale:
            by_locale.setdefault(locale, set()).update(names)
        else:
            unmapped.add(country)
            by_locale.setdefault("other", set()).update(names)
    return by_locale, unmapped


def merge_locale_dicts(base: dict, overlay: dict) -> dict:
    """Merge two locale->set dicts, combining sets for matching locales."""
    merged = {}
    for locale in set(list(base.keys()) + list(overlay.keys())):
        merged[locale] = set()
        if locale in base:
            merged[locale].update(base[locale])
        if locale in overlay:
            merged[locale].update(overlay[locale])
    return merged


def format_python_dict(locale_names: dict, var_name: str,
                       locale_labels: dict, line_width: int = 99) -> str:
    """Format a locale->names mapping as a Python dict of sets."""
    lines = [f"{var_name} = {{"]
    for locale in sorted(locale_names.keys()):
        names = locale_names[locale]
        if not names:
            continue
        label = locale_labels.get(locale, locale.upper())
        sorted_names = sorted(names)
        lines.append(f'    # {label} ({len(names)} names)')
        lines.append(f'    "{locale}": {{')
        current_line = "        "
        for name in sorted_names:
            entry = f'"{name}", '
            if len(current_line) + len(entry) > line_width:
                lines.append(current_line.rstrip())
                current_line = "        "
            current_line += entry
        if current_line.strip():
            lines.append(current_line.rstrip())
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def format_flat_set(locale_names: dict, var_name: str, line_width: int = 99) -> str:
    """Format all names across locales as a single flat set."""
    all_names = set()
    for names in locale_names.values():
        all_names.update(names)
    sorted_names = sorted(all_names)
    lines = [f"{var_name} = {{"]
    current_line = "    "
    for name in sorted_names:
        entry = f'"{name}", '
        if len(current_line) + len(entry) > line_width:
            lines.append(current_line.rstrip())
            current_line = "    "
        current_line += entry
    if current_line.strip():
        lines.append(current_line.rstrip())
    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ingest and unify name data")
    parser.add_argument("--local", action="store_true", help="Use cached local CSVs")
    parser.add_argument("--stats-only", action="store_true", help="Just print stats")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write generated Python code to file")
    args = parser.parse_args()

    # Load or download CSVs
    if args.local:
        if not FORENAMES_FILE.exists() or not SURNAMES_FILE.exists():
            print("ERROR: Local CSVs not found. Run without --local first to download.")
            sys.exit(1)
        print("Loading cached CSVs...")
        forenames_content = load_csv(FORENAMES_FILE)
        surnames_content = load_csv(SURNAMES_FILE)
    else:
        print("Downloading CSVs from GitHub...")
        forenames_content = download_csv(FORENAMES_URL, FORENAMES_FILE)
        surnames_content = download_csv(SURNAMES_URL, SURNAMES_FILE)

    # Parse popular names
    print("\nParsing popular forenames...")
    forenames_by_country = parse_forenames(forenames_content)
    popular_forenames_by_locale, unmapped_fn = group_by_locale(forenames_by_country)
    pop_first_total = sum(len(v) for v in popular_forenames_by_locale.values())
    print(f"  {pop_first_total} forenames from {len(forenames_by_country)} countries "
          f"-> {len(popular_forenames_by_locale)} locales")

    print("\nParsing popular surnames...")
    surnames_by_country = parse_surnames(surnames_content)
    popular_surnames_by_locale, unmapped_sn = group_by_locale(surnames_by_country)
    pop_last_total = sum(len(v) for v in popular_surnames_by_locale.values())
    print(f"  {pop_last_total} surnames from {len(surnames_by_country)} countries "
          f"-> {len(popular_surnames_by_locale)} locales")

    all_unmapped = unmapped_fn | unmapped_sn
    if all_unmapped:
        print(f"  Unmapped country codes (under 'other'): {sorted(all_unmapped)}")

    # Load curated names
    print("\nLoading curated names from names_database.py...")
    curated_first, curated_last = get_curated_names()
    cur_first_total = sum(len(v) for v in curated_first.values())
    cur_last_total = sum(len(v) for v in curated_last.values())
    print(f"  {cur_first_total} curated forenames across {len(curated_first)} locales")
    print(f"  {cur_last_total} curated surnames across {len(curated_last)} locales")

    # Merge: curated + popular
    print("\nMerging curated + popular names...")
    merged_first = merge_locale_dicts(curated_first, popular_forenames_by_locale)
    merged_last = merge_locale_dicts(curated_last, popular_surnames_by_locale)

    total_first = sum(len(v) for v in merged_first.values())
    total_last = sum(len(v) for v in merged_last.values())
    print(f"  Merged forenames: {total_first} across {len(merged_first)} locales")
    print(f"  Merged surnames:  {total_last} across {len(merged_last)} locales")

    # Stats breakdown
    print(f"\n{'='*60}")
    print(f"UNIFIED NAME DATABASE")
    print(f"{'='*60}")
    print(f"\nForenames by locale:")
    for locale in sorted(merged_first.keys()):
        label = LOCALE_NAMES.get(locale, locale)
        count = len(merged_first[locale])
        curated_count = len(curated_first.get(locale, set()))
        popular_count = len(popular_forenames_by_locale.get(locale, set()))
        print(f"  {locale:6s} ({label:15s}): {count:4d} names "
              f"(curated: {curated_count:3d}, popular: {popular_count:3d})")

    print(f"\nSurnames by locale:")
    for locale in sorted(merged_last.keys()):
        label = LOCALE_NAMES.get(locale, locale)
        count = len(merged_last[locale])
        curated_count = len(curated_last.get(locale, set()))
        popular_count = len(popular_surnames_by_locale.get(locale, set()))
        print(f"  {locale:6s} ({label:15s}): {count:4d} names "
              f"(curated: {curated_count:3d}, popular: {popular_count:3d})")

    if args.stats_only:
        return

    # Generate unified Python code
    print("\nGenerating unified Python code...")
    code_lines = [
        '"""',
        "Unified names database organized by locale.",
        "",
        "Sources:",
        "  - Curated names (manually maintained in names_database.py)",
        "  - github.com/sigpwned/popular-names-by-country-dataset (CC0 license)",
        "",
        "Generated by tools/ingest_popular_names.py",
        '"""',
        "",
        "",
        "# ============================================================================",
        "# Forenames by locale (ISO 639-1 language code -> set of lowercase names)",
        "# ============================================================================",
        "",
    ]
    code_lines.append(format_python_dict(merged_first, "FORENAMES_BY_LOCALE", LOCALE_NAMES))
    code_lines.append("")
    code_lines.append("")
    code_lines.append("# ============================================================================")
    code_lines.append("# Surnames by locale (ISO 639-1 language code -> set of lowercase names)")
    code_lines.append("# ============================================================================")
    code_lines.append("")
    code_lines.append(format_python_dict(merged_last, "SURNAMES_BY_LOCALE", LOCALE_NAMES))
    code_lines.append("")
    code_lines.append("")
    code_lines.append("# ============================================================================")
    code_lines.append("# Flat sets (all locales combined) for NamesDatabase")
    code_lines.append("# ============================================================================")
    code_lines.append("")
    code_lines.append(format_flat_set(merged_first, "ALL_FIRST_NAMES"))
    code_lines.append("")
    code_lines.append(format_flat_set(merged_last, "ALL_LAST_NAMES"))
    code_lines.append("")

    generated = "\n".join(code_lines)

    output_path = args.output or (DATA_DIR / "generated_popular_names.py")
    output_path.write_text(generated, encoding="utf-8")
    print(f"\nGenerated: {output_path}")
    print(f"  {total_first} forenames + {total_last} surnames = "
          f"{total_first + total_last} total names")
    print(f"  {len(merged_first)} forename locales, {len(merged_last)} surname locales")


if __name__ == "__main__":
    main()
