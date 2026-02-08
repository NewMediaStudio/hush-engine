#!/usr/bin/env python3
"""
Major Companies Database for Hush Engine

A curated database of major companies for ORGANIZATION detection.
Includes S&P 500 companies, Fortune 500, and other major global corporations.

Coverage: Major US and international corporations
Estimated coverage: 80%+ of company references in typical documents

Usage:
    from hush_engine.data.companies_database import CompaniesDatabase
    db = CompaniesDatabase()
    db.is_company("Apple")  # True
    db.is_company("Microsoft")  # True
"""

from typing import Set, Optional


# ============================================================================
# S&P 500 Companies (as of 2024)
# Includes common name variations and abbreviations
# ============================================================================

SP500_COMPANIES = {
    # Technology
    "apple", "microsoft", "amazon", "alphabet", "google", "meta", "facebook",
    "nvidia", "tesla", "broadcom", "oracle", "accenture", "adobe", "salesforce",
    "cisco", "intel", "ibm", "qualcomm", "intuit", "amd", "texas instruments",
    "servicenow", "applied materials", "analog devices", "lam research", "synopsys",
    "cadence", "autodesk", "workday", "palo alto networks", "crowdstrike", "fortinet",
    "datadog", "snowflake", "mongodb", "splunk", "vmware", "dell", "hp", "hpe",
    "micron", "western digital", "seagate", "netapp", "arista", "motorola",
    "paypal", "block", "square", "fiserv", "fis", "global payments", "paychex",

    # Financial Services
    "jpmorgan", "jp morgan", "chase", "bank of america", "wells fargo", "citigroup",
    "citi", "goldman sachs", "morgan stanley", "blackrock", "charles schwab",
    "american express", "amex", "visa", "mastercard", "capital one", "pnc",
    "us bancorp", "truist", "state street", "northern trust", "bny mellon",
    "raymond james", "ameriprise", "principal", "t. rowe price", "invesco",
    "franklin templeton", "aon", "marsh", "willis towers watson", "metlife",
    "prudential", "aflac", "allstate", "progressive", "travelers", "chubb",
    "hartford", "aig", "lincoln national", "unum", "voya", "synchrony",

    # Healthcare & Pharmaceuticals
    "unitedhealth", "johnson & johnson", "j&j", "eli lilly", "lilly", "abbvie",
    "merck", "pfizer", "thermo fisher", "abbott", "danaher", "bristol-myers",
    "amgen", "gilead", "regeneron", "vertex", "moderna", "biogen", "illumina",
    "intuitive surgical", "stryker", "medtronic", "boston scientific", "edwards",
    "zimmer biomet", "becton dickinson", "baxter", "cvs", "cigna", "elevance",
    "anthem", "humana", "centene", "molina", "hca", "quest diagnostics", "labcorp",
    "idexx", "veeva", "iqvia", "mckesson", "cardinal health", "amerisourcebergen",

    # Consumer & Retail
    "walmart", "costco", "home depot", "target", "lowes", "lowe's", "dollar general",
    "dollar tree", "ross stores", "tjx", "ulta", "best buy", "autozone", "o'reilly",
    "advance auto", "tractor supply", "bath & body works", "gap", "nordstrom",
    "macy's", "kohl's", "dillard's", "williams-sonoma", "etsy", "ebay", "chewy",
    "wayfair", "nike", "lululemon", "vf corporation", "ralph lauren", "pvh",
    "tapestry", "capri", "hasbro", "mattel", "estee lauder", "clorox", "church & dwight",
    "coca-cola", "pepsi", "pepsico", "keurig", "monster beverage", "constellation",
    "brown-forman", "molson coors", "mcdonald's", "mcdonalds", "starbucks", "chipotle",
    "yum brands", "darden", "domino's", "papa john's", "dunkin", "wendy's",
    "procter & gamble", "p&g", "colgate", "kimberly-clark", "general mills",
    "kellogg", "kraft heinz", "mondelez", "hershey", "campbell", "j.m. smucker",
    "conagra", "hormel", "tyson", "pilgrim's pride", "sysco", "kroger", "albertsons",

    # Industrial & Manufacturing
    "3m", "honeywell", "caterpillar", "deere", "john deere", "parker hannifin",
    "illinois tool works", "itw", "emerson", "rockwell", "eaton", "ge", "general electric",
    "boeing", "lockheed martin", "raytheon", "northrop grumman", "general dynamics",
    "l3harris", "textron", "leidos", "saic", "huntington ingalls", "bae systems",
    "united technologies", "carrier", "otis", "trane", "johnson controls", "paccar",
    "cummins", "dover", "xylem", "pentair", "graco", "lincoln electric", "stanley black",
    "snap-on", "fastenal", "w.w. grainger", "grainger", "msc industrial", "cintas",
    "unifirst", "republic services", "waste management", "rollins", "copart", "lkq",

    # Energy & Utilities
    "exxon", "exxonmobil", "chevron", "conocophillips", "schlumberger", "slb",
    "halliburton", "baker hughes", "marathon petroleum", "valero", "phillips 66",
    "occidental", "eog", "pioneer", "devon", "diamondback", "coterra", "hess",
    "marathon oil", "apa", "apache", "kinder morgan", "williams", "oneok", "targa",
    "cheniere", "nextera", "duke energy", "southern company", "dominion", "sempra",
    "pge", "pg&e", "edison", "xcel", "entergy", "firstenergy", "dte", "ameren",
    "cms energy", "wec", "evergy", "alliant", "atmos", "nisource", "centerpoint",

    # Media & Entertainment
    "disney", "walt disney", "netflix", "comcast", "warner bros", "paramount",
    "fox", "news corp", "live nation", "madison square garden", "msg", "spotify",
    "roku", "electronic arts", "ea", "activision", "take-two", "roblox", "unity",
    "match group", "pinterest", "snap", "twitter", "linkedin", "yelp", "tripadvisor",

    # Telecommunications
    "at&t", "verizon", "t-mobile", "lumen", "frontier", "charter", "comcast",

    # Real Estate
    "prologis", "american tower", "equinix", "crown castle", "digital realty",
    "public storage", "welltower", "ventas", "healthpeak", "realty income",
    "simon property", "boston properties", "alexandria real estate", "avalonbay",
    "equity residential", "essex property", "invitation homes", "camden property",
    "mid-america apartment", "sun communities", "extra space", "cubesmart",
    "iron mountain", "weyerhaeuser", "rayonier", "potlatchdeltic", "regency centers",
    "federal realty", "kimco", "national retail", "brixmor", "vornado", "sl green",

    # Transportation & Logistics
    "union pacific", "csx", "norfolk southern", "ups", "fedex", "old dominion",
    "jb hunt", "xpo logistics", "expeditors", "ch robinson", "uber", "lyft",
    "delta", "united airlines", "american airlines", "southwest", "alaska air",
    "jetblue", "carnival", "royal caribbean", "norwegian cruise",

    # Materials & Chemicals
    "linde", "air products", "dow", "dupont", "lyondellbasell", "ppg", "sherwin-williams",
    "ecolab", "nucor", "steel dynamics", "cleveland-cliffs", "freeport-mcmoran",
    "newmont", "mosaic", "cf industries", "corteva", "fmc", "albemarle", "celanese",
    "eastman", "westrock", "packaging corp", "international paper", "sonoco", "avery dennison",
}

# ============================================================================
# Fortune 500 / Major International Companies (additions to S&P 500)
# ============================================================================

MAJOR_INTERNATIONAL_COMPANIES = {
    # Tech Giants
    "samsung", "sony", "panasonic", "lg", "toshiba", "hitachi", "fujitsu", "nec",
    "alibaba", "tencent", "baidu", "jd.com", "bytedance", "tiktok", "huawei", "xiaomi",
    "lenovo", "asus", "acer", "tsmc", "foxconn", "sap", "siemens", "ericsson", "nokia",

    # Automotive
    "toyota", "volkswagen", "vw", "mercedes", "mercedes-benz", "daimler", "bmw",
    "ford", "general motors", "gm", "chrysler", "stellantis", "honda", "nissan",
    "hyundai", "kia", "mazda", "subaru", "mitsubishi", "suzuki", "volvo", "jaguar",
    "land rover", "audi", "porsche", "ferrari", "lamborghini", "bentley", "rolls-royce",
    "rivian", "lucid", "polestar", "nio", "xpeng", "li auto", "byd",

    # Banking & Finance (International)
    "hsbc", "barclays", "ubs", "credit suisse", "deutsche bank", "bnp paribas",
    "societe generale", "santander", "ing", "bbva", "standard chartered", "rbc",
    "td bank", "scotia bank", "manulife", "sun life", "zurich", "allianz", "axa",
    "swiss re", "munich re", "tokio marine", "nomura", "mizuho", "sumitomo mitsui",

    # Consumer Goods (International)
    "nestle", "unilever", "danone", "l'oreal", "lvmh", "kering", "hermes", "burberry",
    "prada", "gucci", "chanel", "dior", "louis vuitton", "rolex", "omega", "cartier",
    "swatch", "ikea", "h&m", "zara", "inditex", "adidas", "puma", "reebok",
    "under armour", "new balance", "asics", "red bull", "heineken", "anheuser-busch",
    "ab inbev", "diageo", "pernod ricard", "bacardi", "beam suntory", "carlsberg",

    # Pharma & Healthcare (International)
    "novartis", "roche", "sanofi", "glaxosmithkline", "gsk", "astrazeneca",
    "novo nordisk", "bayer", "takeda", "teva", "boehringer", "merck kgaa",

    # Energy & Mining (International)
    "shell", "bp", "totalenergies", "total", "equinor", "petrobras", "saudi aramco",
    "aramco", "gazprom", "rosneft", "lukoil", "eni", "repsol", "bhp", "rio tinto",
    "vale", "anglo american", "glencore", "barrick gold", "fortescue",

    # Conglomerates
    "berkshire hathaway", "berkshire", "general electric", "3m", "honeywell",
    "siemens", "philips", "mitsubishi", "mitsui", "marubeni", "itochu", "sumitomo",
    "tata", "reliance", "adani", "softbank", "rakuten",

    # Consulting & Professional Services
    "mckinsey", "boston consulting", "bcg", "bain", "deloitte", "pwc",
    "pricewaterhousecoopers", "kpmg", "ey", "ernst & young", "accenture",
    "capgemini", "cognizant", "infosys", "wipro", "tcs", "hcl", "tech mahindra",

    # Airlines (International)
    "lufthansa", "air france", "klm", "british airways", "virgin atlantic",
    "emirates", "qatar airways", "singapore airlines", "cathay pacific",
    "qantas", "air canada", "japan airlines", "ana", "korean air", "thai airways",
}

# ============================================================================
# Common company name words (for context detection)
# ============================================================================

COMPANY_CONTEXT_WORDS = {
    "corporation", "corp", "incorporated", "inc", "limited", "ltd", "llc",
    "company", "co", "group", "holdings", "partners", "associates", "enterprises",
    "industries", "international", "global", "worldwide", "solutions", "services",
    "technologies", "systems", "networks", "communications", "consulting",
    "capital", "investments", "ventures", "management", "healthcare", "pharma",
    "energy", "resources", "materials", "financial", "insurance", "bank", "trust",
}

# ============================================================================
# All Companies Combined
# ============================================================================

ALL_COMPANIES: Set[str] = SP500_COMPANIES | MAJOR_INTERNATIONAL_COMPANIES


class CompaniesDatabase:
    """
    Database for looking up major companies.

    Provides fast O(1) lookup for company names to support ORGANIZATION detection.
    """

    def __init__(self):
        """Initialize the companies database."""
        # Normalize all companies to lowercase for case-insensitive matching
        self._all_companies = {company.lower() for company in ALL_COMPANIES}
        self._sp500 = {company.lower() for company in SP500_COMPANIES}
        self._context_words = {word.lower() for word in COMPANY_CONTEXT_WORDS}

    def is_company(self, text: str) -> bool:
        """
        Check if text is a known major company.

        Args:
            text: The text to check

        Returns:
            True if the text is a known company name
        """
        return text.lower().strip() in self._all_companies

    def is_sp500(self, text: str) -> bool:
        """
        Check if text is a known S&P 500 company.

        Args:
            text: The text to check

        Returns:
            True if the text is a known S&P 500 company name
        """
        return text.lower().strip() in self._sp500

    def is_company_word(self, text: str) -> bool:
        """
        Check if text is a common company context word.

        Args:
            text: The text to check

        Returns:
            True if the text is a company context word
        """
        return text.lower().strip() in self._context_words

    def get_confidence_boost(self, text: str) -> float:
        """
        Get a confidence boost for a company name based on its importance.

        Args:
            text: The company name to check

        Returns:
            A confidence boost value (0.0 if not a company, 0.1-0.2 if a company)
        """
        lower_text = text.lower().strip()

        if lower_text in self._sp500:
            return 0.2  # S&P 500 companies are high confidence
        elif lower_text in self._all_companies:
            return 0.15  # Other major companies

        return 0.0

    @property
    def company_count(self) -> int:
        """Return the number of companies in the database."""
        return len(self._all_companies)


# Global instance for convenience
_companies_db: Optional[CompaniesDatabase] = None


def get_companies_db() -> CompaniesDatabase:
    """Get the global companies database instance."""
    global _companies_db
    if _companies_db is None:
        _companies_db = CompaniesDatabase()
    return _companies_db
