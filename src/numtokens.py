"""Number tokens in titles (sizes, capacities, pack counts) and the variant check.

"500 ml" and "1 l" are different variants even when photos and wording look
alike, so a pair whose titles disagree on a quantity of the same kind is
probably not the same product. Units are normalised first, so "1 l" and
"1000 ml" agree.
"""
import re

# unit -> (kind, factor to the base unit of that kind)
UNITS = {
    "ml": ("volume", 1), "l": ("volume", 1000), "ltr": ("volume", 1000), "litre": ("volume", 1000),
    "liter": ("volume", 1000),
    "g": ("mass", 1), "gm": ("mass", 1), "gr": ("mass", 1), "kg": ("mass", 1000),  # "gr": common Shopee spelling
    "mb": ("storage", 1 / 1024), "gb": ("storage", 1), "tb": ("storage", 1024),
    "mm": ("length", 1), "cm": ("length", 10), "m": ("length", 1000), "inch": ("length", 25.4),
    "in": ("length", 25.4), '"': ("length", 25.4),
    "w": ("power", 1), "mah": ("battery", 1),
    "pcs": ("count", 1), "pc": ("count", 1), "pieces": ("count", 1),
}
NUM = r"(\d+(?:[.,]\d+)?)"
# Longest units first so "gb" is not read as "g". "in" is skipped before a digit ("2 in 1").
UNIT_RE = re.compile(NUM + r"\s*(" + "|".join(sorted(map(re.escape, UNITS), key=len, reverse=True))
                     + r")(?![a-z])(?<!in(?=\s*\d))", re.IGNORECASE)
COUNT_RE = re.compile(r"\b(?:pack|set)\s+of\s+(\d+)\b", re.IGNORECASE)
# Indonesian "isi N" = N items inside; skipped when a unit follows ("isi 500ml" is a volume).
ISI_RE = re.compile(r"\bisi\s*(\d+)(?![\d.,]*\s*[a-z\"])", re.IGNORECASE)


def parse_number(s):
    """'1.5' / '1,5' -> 1.5; '1,000' / '1.000' -> 1000.

    A separator followed by exactly three digits is a thousands separator
    (Indonesian titles write 1.000); otherwise it is a decimal point.
    """
    m = re.fullmatch(r"(\d+)[.,](\d+)", s)
    if m and len(m.group(2)) == 3:
        return float(m.group(1) + m.group(2))
    return float(s.replace(",", "."))


def extract(title):
    """Set of (kind, normalised value) tokens in a title, e.g. {('volume', 1000.0)}."""
    tokens = set()
    for num, unit in UNIT_RE.findall(title):
        kind, factor = UNITS[unit.lower()]
        tokens.add((kind, round(parse_number(num) * factor, 2)))
    for num in COUNT_RE.findall(title) + ISI_RE.findall(title):
        tokens.add(("count", float(num)))
    return tokens


def compare(tokens_a, tokens_b):
    """(match, conflict) flags for two token sets.

    match:    the titles share at least one identical (kind, value) token.
    conflict: some kind appears in both titles, but no value of that kind is shared.
    """
    match = bool(tokens_a & tokens_b)
    shared_kinds = {k for k, _ in tokens_a} & {k for k, _ in tokens_b}
    conflict = any(not ({v for k, v in tokens_a if k == kind} & {v for k, v in tokens_b if k == kind})
                   for kind in shared_kinds)
    return match, conflict
