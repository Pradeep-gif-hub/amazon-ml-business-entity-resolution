"""
High-performance text normalization and tokenization for entity resolution.
Handles multilingual text, accent folding, punctuation cleaning, domain stripping,
nospace compact keys, legal suffix standardization, and address keys.
"""

import re
import unicodedata
from typing import Set, Tuple, List, Optional

# Pre-compiled regex patterns
RE_NOISE_CHARS = re.compile(r"^[-\#\<\>\*\/\(\)\[\]\"\'\:\;]+|[-\#\<\>\*\/\(\)\[\]\"\'\:\;]+$")
RE_PUNCT_SPLIT = re.compile(r"[^\w\s]+")
RE_MULTI_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+[a-zA-Z]?\b")
RE_DOMAIN_SUFFIX = re.compile(r"\b(com|org|net|in|fr|co|io|biz|info|gov|edu|us|tech|online|store|shop)\b")

# Legal suffixes mapping (standardization)
LEGAL_SUFFIXES_MAP = {
    "incorporated": "inc",
    "corporation": "corp",
    "limited liability company": "llc",
    "limited": "ltd",
    "company": "co",
    "private limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "limited liability partnership": "llp",
    "public limited company": "plc",
    "societe par actions simplifiee": "sas",
    "societe anonyme": "sa",
    "societe a responsabilite limitee": "sarl",
    "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "societe civile immobiliere": "sci",
    "etablissements": "ets",
    "compagnie": "cie"
}

# Legal suffixes to remove for core name generation
LEGAL_TERMS_TO_STRIP = {
    "inc", "corp", "corporation", "incorporated", "llc", "ltd", "limited", "co", "company",
    "pvt", "private", "plc", "llp", "sa", "sas", "sarl", "eurl", "sci", "ets", "cie",
    "gmbh", "holding", "holdings", "group", "enterprises", "associates", "consultants",
    "consulting", "services", "solutions", "traders", "trading", "industries", "agency",
    "the", "and", "&", "com", "org", "net", "in", "fr", "io", "co", "biz", "info", "formerly", "dba"
}

# Address abbreviations mapping
ADDRESS_ABBREV_MAP = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "drive": "dr",
    "boulevard": "blvd",
    "lane": "ln",
    "court": "ct",
    "circle": "cir",
    "parkway": "pkwy",
    "highway": "hwy",
    "building": "bldg",
    "floor": "flr",
    "apartment": "apt",
    "suite": "ste",
    "room": "rm",
    "department": "dept",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
    "sector": "sec",
    "nagar": "ngr",
    "colony": "col",
    "marg": "mrg",
    "opposite": "opp",
    "near": "nr",
    "behind": "bhnd",
    "saint": "st",
    "post office box": "po box",
    "p.o. box": "po box",
    "p o box": "po box"
}

def remove_accents(text: str) -> str:
    """Normalize unicode and strip Latin diacritics while preserving non-Latin scripts."""
    if text.isascii():
        return text
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")

def clean_text_basic(text: str) -> str:
    """Basic cleaning: accent removal, lowercasing, whitespace canonicalization."""
    if not text:
        return ""
    text = remove_accents(text.lower())
    text = RE_PUNCT_SPLIT.sub(" ", text)
    return RE_MULTI_WHITESPACE.sub(" ", text).strip()

def normalize_business_name(name: str) -> str:
    """Normalize a business name into a standard comparable form."""
    if not name:
        return ""
    name = RE_NOISE_CHARS.sub("", name.strip())
    clean = clean_text_basic(name)
    tokens = clean.split()
    if not tokens:
        return ""
    
    res_tokens = []
    for t in tokens:
        res_tokens.append(LEGAL_SUFFIXES_MAP.get(t, t))
    return " ".join(res_tokens)

def extract_core_name(name: str) -> str:
    """Extract core business name with legal entity suffixes and stop terms removed."""
    clean = normalize_business_name(name)
    if not clean:
        return ""
    tokens = clean.split()
    core_tokens = [t for t in tokens if t not in LEGAL_TERMS_TO_STRIP and len(t) > 1]
    if not core_tokens:
        return clean
    return " ".join(core_tokens)

def extract_nospace_name(core_name: str) -> str:
    """Extract compact alphanumeric name for concatenated domains / typos."""
    if not core_name:
        return ""
    return re.sub(r"[^a-z0-9]", "", core_name)

def extract_name_tokens(name: str, min_len: int = 2) -> Set[str]:
    """Extract set of meaningful name tokens."""
    clean = normalize_business_name(name)
    if not clean:
        return set()
    return {t for t in clean.split() if len(t) >= min_len and t not in LEGAL_TERMS_TO_STRIP}

def normalize_business_address(addr: str) -> str:
    """Normalize business address with standardized street words and noise stripping."""
    if not addr:
        return ""
    clean = clean_text_basic(addr)
    if not clean:
        return ""
    tokens = clean.split()
    res_tokens = []
    for t in tokens:
        res_tokens.append(ADDRESS_ABBREV_MAP.get(t, t))
    return " ".join(res_tokens)

def extract_address_numbers(addr: str) -> Set[str]:
    """Extract house numbers, postal/PIN codes, unit numbers from address."""
    if not addr:
        return set()
    return set(RE_NUMBERS.findall(addr.lower()))

def extract_address_tokens(addr: str, min_len: int = 3) -> Set[str]:
    """Extract set of significant address tokens (cities, localities, street names)."""
    clean = normalize_business_address(addr)
    if not clean:
        return set()
    stop_addr_terms = {"st", "rd", "ave", "dr", "blvd", "ln", "ct", "apt", "ste", "flr", "bldg", "usa", "india", "france"}
    return {t for t in clean.split() if len(t) >= min_len and t not in stop_addr_terms}

def extract_address_street_keys(addr: str) -> List[str]:
    """Extract (number, street_token) combinations for high-precision address blocking."""
    clean = normalize_business_address(addr)
    if not clean:
        return []
    numbers = extract_address_numbers(addr)
    tokens = extract_address_tokens(addr)
    keys = []
    for num in numbers:
        for t in tokens:
            if not t.isdigit() and len(t) >= 4:
                keys.append(f"{num}_{t}")
    return keys

class RecordProfile:
    """Compact pre-computed profile of an entity record for fast matching and blocking."""
    __slots__ = ("entity_id", "country", "raw_name", "raw_addr", "clean_name", 
                 "core_name", "nospace_name", "name_tokens", "clean_addr", 
                 "addr_numbers", "addr_tokens", "addr_street_keys")
    
    def __init__(self, entity_id: str, business_name: str, business_address: str, country: str):
        self.entity_id = entity_id
        self.country = country
        self.raw_name = business_name
        self.raw_addr = business_address
        self.clean_name = normalize_business_name(business_name)
        self.core_name = extract_core_name(business_name)
        self.nospace_name = extract_nospace_name(self.core_name)
        self.name_tokens = extract_name_tokens(business_name)
        self.clean_addr = normalize_business_address(business_address)
        self.addr_numbers = extract_address_numbers(business_address)
        self.addr_tokens = extract_address_tokens(business_address)
        self.addr_street_keys = extract_address_street_keys(business_address)
