#!/usr/bin/env python3
"""
Negative Gazetteer for PII Detection False Positive Suppression

Contains common words and phrases that should NOT be detected as PII,
organized by entity type. This reduces false positives for PERSON, COMPANY,
and LOCATION entities.

Coverage:
- ~500 common English words that cause PERSON false positives
- Entity-specific denylists for PERSON, COMPANY, LOCATION
- Software/brand names, UI elements, generic business terms

Usage:
    from hush_engine.data.negative_gazetteer import is_negative_match

    if is_negative_match("Adobe", "PERSON"):
        # Skip this detection - it's a false positive
        pass
"""

from typing import Set, Dict, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Common English Words That Cause PERSON False Positives
# ============================================================================

# Words that are both common English words AND sometimes names
# These should be filtered unless they have strong name context
COMMON_ENGLISH_WORDS: Set[str] = {
    # Common words that are also names (high FP rate)
    "will", "bill", "jack", "rose", "grace", "hope", "faith", "joy",
    "amber", "jade", "holly", "ivy", "ruby", "pearl", "april", "may",
    "june", "summer", "autumn", "winter", "spring", "dawn", "eve",
    "chance", "chase", "hunter", "mason", "carter", "cooper", "tucker",
    "fisher", "archer", "walker", "parker", "spencer", "porter", "sawyer",
    "taylor", "tyler", "jordan", "morgan", "logan", "ryan", "austin",
    "dallas", "phoenix", "denver", "paris", "london", "brooklyn", "india",
    "asia", "china", "kenya", "geneva", "carolina", "georgia", "virginia",
    "montana", "nevada", "dakota", "savannah", "sierra", "aurora",
    "melody", "harmony", "destiny", "trinity", "serenity", "felicity",
    "charity", "prudence", "patience", "honor", "justice", "liberty",
    "angel", "saint", "bishop", "dean", "pastor", "king", "queen",
    "prince", "duke", "earl", "baron", "knight", "major", "general",
    "judge", "page", "grant", "wade", "lane", "dale", "glen", "heath",
    "brook", "lake", "river", "stone", "storm", "rain", "snow", "frost",
    "sky", "cloud", "star", "moon", "sunny", "cliff", "ridge", "peak",
    "forest", "field", "meadow", "grove", "wood", "moss", "fern", "reed",
    "reed", "bush", "branch", "thorn", "berry", "cherry", "olive", "ginger",
    "basil", "sage", "rosemary", "hazel", "willow", "laurel", "heather",

    # Months (commonly detected as names)
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",

    # Days
    "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",

    # Colors that are names
    "violet", "scarlett", "ruby", "jade", "amber", "ebony", "ivory",
    "sienna", "olive", "hazel", "coral", "rose", "lilac", "fern",

    # Occupations/titles often detected as names
    "baker", "butler", "carpenter", "cook", "farmer", "fisher", "gardner",
    "hunter", "mason", "miller", "porter", "potter", "shepherd", "smith",
    "taylor", "turner", "walker", "weaver", "archer", "chandler", "draper",
    "fletcher", "fowler", "glover", "mercer", "thatcher", "wheeler", "wright",

    # Generic role words
    "customer", "client", "patient", "member", "agent", "user", "owner",
    "manager", "director", "president", "chairman", "secretary", "treasurer",
    "officer", "executive", "administrator", "coordinator", "supervisor",
    "assistant", "associate", "consultant", "analyst", "specialist",
    "representative", "delegate", "employee", "employer", "applicant",
    "candidate", "recipient", "beneficiary", "subscriber", "contributor",

    # UI/Action words
    "submit", "cancel", "confirm", "delete", "save", "edit", "update",
    "refresh", "reload", "reset", "clear", "close", "open", "view",
    "search", "filter", "sort", "select", "click", "tap", "press",
    "enter", "exit", "login", "logout", "signin", "signup", "register",

    # Status words
    "active", "inactive", "pending", "approved", "rejected", "denied",
    "confirmed", "completed", "processing", "loading", "waiting",

    # Document/form words
    "summary", "details", "notes", "comments", "history", "record",
    "document", "attachment", "signature", "required", "optional",

    # Common adverbs/adjectives detected as names
    "best", "better", "new", "old", "true", "false", "good", "bad",
    "smart", "rich", "wise", "noble", "royal", "golden", "silver",
}


# ============================================================================
# Entity-Specific Negative Gazetteers
# ============================================================================

ENTITY_NEGATIVE_GAZETTEERS: Dict[str, Set[str]] = {
    "PERSON": {
        # Software/Brand names often detected as PERSON
        "adobe", "amazon", "apple", "google", "microsoft", "oracle",
        "tesla", "uber", "lyft", "slack", "zoom", "figma", "notion",
        "dropbox", "spotify", "netflix", "airbnb", "stripe", "twilio",
        "shopify", "salesforce", "atlassian", "github", "gitlab", "jira",
        "trello", "asana", "monday", "hubspot", "zendesk", "intercom",
        "mailchimp", "canva", "sketch", "invision", "zeplin", "abstract",
        "framer", "webflow", "squarespace", "wix", "wordpress", "drupal",
        "magento", "shopware", "prestashop", "bigcommerce", "woocommerce",
        "photoshop", "illustrator", "indesign", "premiere", "aftereffects",
        "lightroom", "acrobat", "reader", "firefox", "chrome", "safari",
        "edge", "opera", "brave", "vivaldi", "tor", "duckduckgo",

        # Tech products/services
        "iphone", "ipad", "macbook", "airpods", "homepod", "appletv",
        "android", "pixel", "galaxy", "oneplus", "huawei", "xiaomi",
        "windows", "linux", "ubuntu", "debian", "fedora", "centos",
        "kubernetes", "docker", "jenkins", "terraform", "ansible", "puppet",
        "elasticsearch", "mongodb", "postgres", "mysql", "redis", "kafka",
        "nginx", "apache", "tomcat", "nodejs", "python", "javascript",

        # Food items commonly detected as names
        "caesar", "napoleon", "benedict", "wellington", "margherita",
        "alfredo", "carbonara", "bolognese", "parmesan", "mozzarella",
        "cheddar", "brie", "camembert", "gruyere", "gouda", "feta",

        # Nationalities/demonyms
        "american", "british", "canadian", "australian", "french", "german",
        "spanish", "italian", "chinese", "japanese", "korean", "indian",
        "mexican", "brazilian", "russian", "african", "european", "asian",

        # Document/form labels
        "name", "first name", "last name", "full name", "given name",
        "surname", "middle name", "maiden name", "nickname", "alias",
        "author", "sender", "recipient", "addressee", "cc", "bcc",

        # Generic descriptors
        "anonymous", "unknown", "unnamed", "redacted", "confidential",
        "private", "public", "personal", "individual", "entity",
    },

    "COMPANY": {
        # Generic business terms (not specific company names)
        "department", "division", "section", "branch", "office",
        "committee", "board", "council", "commission", "agency",
        "bureau", "authority", "administration", "organization",
        "corporation", "enterprise", "establishment", "institution",
        "association", "federation", "union", "alliance", "coalition",
        "partnership", "consortium", "syndicate", "conglomerate",
        "subsidiary", "affiliate", "parent", "holding", "group",

        # Generic service descriptions
        "services", "solutions", "systems", "technologies", "industries",
        "consulting", "advisors", "partners", "associates", "experts",
        "professionals", "specialists", "practitioners", "providers",
        "suppliers", "vendors", "contractors", "developers", "builders",

        # Departments/functions
        "hr", "human resources", "it", "information technology",
        "r&d", "research and development", "qa", "quality assurance",
        "marketing", "sales", "finance", "accounting", "legal",
        "operations", "logistics", "procurement", "support", "service",

        # Legal/regulatory terms
        "plaintiff", "defendant", "petitioner", "respondent", "appellant",
        "applicant", "claimant", "complainant", "prosecutor", "witness",

        # Document labels
        "company", "business", "employer", "vendor", "supplier",
        "customer", "client", "account", "merchant", "retailer",
    },

    "LOCATION": {
        # Generic place words
        "location", "address", "place", "area", "region", "zone",
        "site", "venue", "destination", "origin", "route", "path",
        "district", "territory", "sector", "quarter", "block",

        # Abstract/relative locations
        "here", "there", "somewhere", "anywhere", "everywhere", "nowhere",
        "nearby", "remote", "local", "global", "worldwide", "nationwide",
        "statewide", "citywide", "regional", "national", "international",

        # Directional terms
        "north", "south", "east", "west", "northeast", "northwest",
        "southeast", "southwest", "central", "northern", "southern",
        "eastern", "western", "upper", "lower", "inner", "outer",

        # Generic building/structure terms
        "building", "office", "headquarters", "branch", "facility",
        "warehouse", "factory", "plant", "station", "terminal", "depot",
        "center", "centre", "complex", "campus", "park", "plaza",

        # Common field labels
        "street address", "mailing address", "billing address",
        "shipping address", "home address", "work address",
        "permanent address", "temporary address", "current address",

        # Preposition phrases (incomplete addresses)
        "in the", "at the", "on the", "near the", "by the",
        "within the", "outside the", "around the", "across the",
    },

    "ADDRESS": {
        # Inherits from LOCATION plus address-specific terms
        "po box", "p.o. box", "post office box", "mailbox",
        "apt", "apartment", "unit", "suite", "ste", "floor", "fl",
        "room", "rm", "building", "bldg", "tower", "wing",

        # Address field labels
        "street", "avenue", "road", "drive", "lane", "court",
        "boulevard", "way", "place", "circle", "terrace", "trail",
        "highway", "route", "parkway", "expressway", "freeway",
    },
}


# ============================================================================
# API Functions
# ============================================================================

def is_negative_match(text: str, entity_type: str) -> bool:
    """
    Check if text is in the negative gazetteer for the given entity type.

    Args:
        text: The detected text to check
        entity_type: Type of PII entity (PERSON, COMPANY, LOCATION, ADDRESS)

    Returns:
        True if the text should be filtered as a false positive
    """
    text_lower = text.lower().strip()

    # Check common words first (applies to all entity types)
    if text_lower in COMMON_ENGLISH_WORDS:
        return True

    # Check entity-specific gazetteer
    if entity_type in ENTITY_NEGATIVE_GAZETTEERS:
        if text_lower in ENTITY_NEGATIVE_GAZETTEERS[entity_type]:
            return True

    # For ADDRESS, also check LOCATION gazetteer
    if entity_type == "ADDRESS":
        if text_lower in ENTITY_NEGATIVE_GAZETTEERS.get("LOCATION", set()):
            return True

    return False


def is_single_common_word(text: str, entity_type: str) -> bool:
    """
    Check if text is a single common word without corporate suffix.

    For COMPANY entities, single common words like "Apple" or "Target"
    need a corporate suffix (Inc, Ltd, Corp) to be considered valid.

    Args:
        text: The detected text
        entity_type: Type of PII entity

    Returns:
        True if it's a single common word without sufficient context
    """
    if entity_type != "COMPANY":
        return False

    text_lower = text.lower().strip()

    # Check if single token
    if ' ' in text_lower:
        return False

    # Check if common word
    if text_lower not in COMMON_ENGLISH_WORDS:
        return False

    # Check for corporate suffix in original text
    corporate_suffixes = ["inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.",
                          "co", "co.", "plc", "gmbh", "ag", "sa", "nv", "bv"]
    text_parts = text.lower().split()
    if any(suffix in text_parts for suffix in corporate_suffixes):
        return False

    return True


def get_negative_words_for_entity(entity_type: str) -> Set[str]:
    """
    Get the complete set of negative words for an entity type.

    Combines common words with entity-specific gazetteer.

    Args:
        entity_type: Type of PII entity

    Returns:
        Set of words that should be filtered for this entity type
    """
    result = COMMON_ENGLISH_WORDS.copy()

    if entity_type in ENTITY_NEGATIVE_GAZETTEERS:
        result.update(ENTITY_NEGATIVE_GAZETTEERS[entity_type])

    if entity_type == "ADDRESS":
        result.update(ENTITY_NEGATIVE_GAZETTEERS.get("LOCATION", set()))

    return result


def load_feedback_false_positives(feedback_path: Path) -> Dict[str, Set[str]]:
    """
    Load false positives from feedback files to extend gazetteer.

    Reads JSON feedback files and extracts text that was marked as
    false positive (detected but shouldn't have been).

    Args:
        feedback_path: Path to feedback directory

    Returns:
        Dict mapping entity types to sets of false positive texts
    """
    updates: Dict[str, Set[str]] = {}

    if not feedback_path.exists():
        logger.debug(f"Feedback path does not exist: {feedback_path}")
        return updates

    try:
        for f in feedback_path.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    entry = json.load(fp)

                detected_type = entry.get("detectedEntityType", "")
                suggested_types = entry.get("suggestedEntityTypes", [])
                detected_text = entry.get("detectedText", "").lower().strip()

                # If suggested types is empty or doesn't include detected type,
                # this is a false positive
                if detected_text and (not suggested_types or detected_type not in suggested_types):
                    if detected_type not in updates:
                        updates[detected_type] = set()
                    updates[detected_type].add(detected_text)

            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Skipping invalid feedback file {f}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Error loading feedback false positives: {e}")

    return updates


def get_gazetteer_stats() -> Dict[str, int]:
    """
    Get statistics about the negative gazetteer.

    Returns:
        Dict with counts for each category
    """
    stats = {
        "common_english_words": len(COMMON_ENGLISH_WORDS),
        "person_negatives": len(ENTITY_NEGATIVE_GAZETTEERS.get("PERSON", set())),
        "company_negatives": len(ENTITY_NEGATIVE_GAZETTEERS.get("COMPANY", set())),
        "location_negatives": len(ENTITY_NEGATIVE_GAZETTEERS.get("LOCATION", set())),
        "address_negatives": len(ENTITY_NEGATIVE_GAZETTEERS.get("ADDRESS", set())),
    }
    stats["total"] = sum(stats.values())
    return stats
