"""Text normalisation, applied before any lookup.

Order matters and matches the rules doc:
  1. Turkish-aware uppercasing (i -> İ, ı -> I)
  2. Strip known budget/legal suffixes like "(632 SKHK)" or "(ÖZ GELİR)"
  3. Expand abbreviations ("D. BŞK." -> "DAİRE BAŞKANLIĞI")
  4. Collapse whitespace and trim repeated trailing letters ("FAKÜLTESİİİ")

Parentheses are not stripped blindly. Some official unit names carry them
(e.g. "İNGİLİZ DİLİ EĞİTİMİ (M.A., PH.D.)"), so only the known suffix
patterns are removed.
"""
import json
import re

from engine.config import get_data_dir

# str.upper() turns "i" into "I", which is wrong for Turkish ("İ").
_TR_LOWER_TO_UPPER = {
    "i": "İ",
    "ı": "I",
    "ş": "Ş",
    "ğ": "Ğ",
    "ü": "Ü",
    "ö": "Ö",
    "ç": "Ç",
}


def turkish_upper(text: str) -> str:
    """Uppercase with Turkish rules for i/ı."""
    if text is None:
        return text
    out = []
    for ch in text:
        if ch in _TR_LOWER_TO_UPPER:
            out.append(_TR_LOWER_TO_UPPER[ch])
        else:
            out.append(ch.upper())
    return "".join(out)


_ABBREVIATIONS_CACHE = {}


def _abbrev_table(data_dir=None) -> dict:
    """Load abbreviations.json once per data dir."""
    data_dir = data_dir or get_data_dir()
    key = str(data_dir)
    if key not in _ABBREVIATIONS_CACHE:
        with open(data_dir / "abbreviations.json", encoding="utf-8") as f:
            raw = json.load(f)
        # People usually drop the dot ("Yo" instead of "YO."), so register
        # the dotless form too.
        expanded = dict(raw)
        for k, v in raw.items():
            if k.endswith("."):
                stripped = k.rstrip(".")
                if stripped and stripped not in expanded:
                    expanded[stripped] = v
        _ABBREVIATIONS_CACHE[key] = expanded
    return _ABBREVIATIONS_CACHE[key]


def expand_abbreviations(text_upper: str, data_dir=None) -> str:
    """Expand abbreviations. Expects already-uppercased text.

    Longest keys first, so "D. BŞK." wins over a shorter "D.".
    """
    table = _abbrev_table(data_dir)
    keys_sorted = sorted(table.keys(), key=len, reverse=True)
    result = text_upper
    for abbr in keys_sorted:
        target = table[abbr]
        # \b doesn't play well with dotted keys and Turkish letters,
        # so use explicit lookarounds instead.
        pattern = re.compile(
            r"(?<![A-ZÇĞİÖŞÜ0-9])" + re.escape(abbr) + r"(?![A-ZÇĞİÖŞÜ0-9])"
        )
        result = pattern.sub(target, result)
    return result


# Budget/staffing suffixes. They don't change which unit it is, so they are
# removed before matching but returned separately so the reason can mention them.
_LEGAL_BUDGET_SUFFIX_RE = re.compile(
    r"\s*\((?:\s*\d*\s*SKHK|ÖZ GELİR|DÖNER SERMAYE[^)]*)\)\s*",
    re.IGNORECASE,
)


def strip_known_legal_suffixes(text: str) -> tuple:
    """Returns (cleaned_text, suffixes_found)."""
    found = _LEGAL_BUDGET_SUFFIX_RE.findall(text)
    cleaned = _LEGAL_BUDGET_SUFFIX_RE.sub(" ", text)
    return cleaned, found


def clean_whitespace(text: str) -> str:
    """Trim and collapse runs of whitespace."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


_TR_I_FAMILY = {"I", "İ"}


def clean_trailing_typo(text: str) -> str:
    """FAKÜLTESİİİ -> FAKÜLTESİ, BAŞKANLIĞIİ -> BAŞKANLIĞI.

    Only touches the last word. The suffix words (FAKÜLTESİ, MÜDÜRLÜĞÜ, ...)
    never end in a double letter, but names in the middle of the text might.
    """
    if not text:
        return text
    parts = text.split(" ")
    last = parts[-1]

    while len(last) >= 2 and last[-1] == last[-2]:
        last = last[:-1]

    # mixed I/İ at the end
    while len(last) >= 2 and last[-1] in _TR_I_FAMILY and last[-2] in _TR_I_FAMILY and last[-1] != last[-2]:
        last = last[:-1]

    parts[-1] = last
    return " ".join(parts)


def normalize(raw_text: str, data_dir=None) -> dict:
    """Run the whole chain.

    Returns {"normalized", "legal_suffixes_found", "steps"}; steps keeps the
    text after each stage, handy when explaining a match.
    """
    if raw_text is None:
        return {"normalized": None, "legal_suffixes_found": [], "steps": []}

    steps = []
    text = raw_text

    text = clean_whitespace(text)
    steps.append(("bosluk_kirpma_1", text))

    text = turkish_upper(text)
    steps.append(("buyuk_harf_tr", text))

    text, legal_suffixes = strip_known_legal_suffixes(text)
    steps.append(("yasal_butce_eki_ayiklama", text))

    text = clean_whitespace(text)
    steps.append(("bosluk_kirpma_2", text))

    text = expand_abbreviations(text, data_dir)
    steps.append(("kisaltma_genisletme", text))

    text = clean_whitespace(text)
    steps.append(("bosluk_kirpma_3", text))

    text = clean_trailing_typo(text)
    steps.append(("trailing_typo_temizligi", text))

    return {
        "normalized": text,
        "legal_suffixes_found": [s.strip() for s in legal_suffixes],
        "steps": steps,
    }


if __name__ == "__main__":
    samples = [
        "bilgi işlem d. bşk.",
        "FAKÜLTESİİİ",
        "MÜHENDİSLİK   FAKÜLTESİ",
        "Döner Sermaye (632 SKHK)",
        "Sağlık Kültür ve Spor Daire Başkanlığı (Öz Gelir)",
        "BAŞKANLIĞIİ",
        "yazılım şb. müd.",
    ]
    for s in samples:
        r = normalize(s)
        print(f"{s!r:55s} -> {r['normalized']!r}  (yasal_ek: {r['legal_suffixes_found']})")
