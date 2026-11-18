"""Org tree helpers and the hierarchy-limited fuzzy search.

Covers two steps of the decision tree:
  step 4 - use the other fields of the record (department, parent unit) to
           narrow the search down to one branch of the tree
  step 6 - fuzzy match inside that branch, without ever crossing a level
           that must not be crossed (a directorate is never a department head
           office, a faculty is never one of its departments)

Similarity is plain difflib so there's nothing extra to install.
"""
import difflib

from engine.canonical_lookup import CanonicalPool
from engine.normalize import normalize, turkish_upper

# Pairs of suffix groups that must never be matched to each other, however
# similar the rest of the name is.
VETO_PAIRS = [
    ({"DAİRE BAŞKANLIĞI", "BAŞKANLIĞI"}, {"MÜDÜRLÜĞÜ"}),
    ({"FAKÜLTESİ"}, {"BÖLÜMÜ", "ANABİLİM DALI"}),
    ({"KOORDİNATÖRLÜĞÜ"}, {"MERKEZİ", "OFİSİ"}),
]

# longest first, so "DAİRE BAŞKANLIĞI" is found before "BAŞKANLIĞI"
STRUCTURAL_SUFFIXES = [
    "DAİRE BAŞKANLIĞI",
    "ANABİLİM DALI",
    "YÜKSEKOKULU",
    "KOORDİNATÖRLÜĞÜ",
    "MÜDÜRLÜĞÜ",
    "FAKÜLTESİ",
    "ENSTİTÜSÜ",
    "BAŞKANLIĞI",
    "BÖLÜMÜ",
    "ŞEFLİĞİ",
    "MERKEZİ",
    "OFİSİ",
    "BİRİMİ",
]

# a missing or extra "VE" shouldn't cost similarity
_CONJUNCTIONS = {"VE", "VEYA", "İLE"}


def structural_class(name_upper: str):
    """Which structural suffix the name ends with, or None."""
    for suf in STRUCTURAL_SUFFIXES:
        if name_upper.endswith(suf):
            return suf
    return None


def violates_veto(candidate_name_upper: str, input_name_upper: str) -> bool:
    """True if candidate and input sit on opposite sides of a veto pair."""
    c_class = structural_class(candidate_name_upper)
    i_class = structural_class(input_name_upper)
    if not c_class or not i_class or c_class == i_class:
        return False
    for side_a, side_b in VETO_PAIRS:
        in_a = c_class in side_a or i_class in side_a
        in_b = c_class in side_b or i_class in side_b
        if in_a and in_b and c_class != i_class:
            a_has_c, a_has_i = c_class in side_a, i_class in side_a
            b_has_c, b_has_i = c_class in side_b, i_class in side_b
            if (a_has_c and b_has_i) or (a_has_i and b_has_c):
                return True
    return False


def _tokens_no_conj(name_upper: str):
    return [t for t in name_upper.split() if t not in _CONJUNCTIONS]


def word_similarity(a_upper: str, b_upper: str) -> float:
    """Word-level similarity (0-1), ignoring conjunctions."""
    ta, tb = _tokens_no_conj(a_upper), _tokens_no_conj(b_upper)
    return difflib.SequenceMatcher(None, ta, tb).ratio()


class HierarchyEngine:
    def __init__(self, pool: CanonicalPool):
        self.pool = pool

    # step 4
    def resolve_parent_unit(self, parent_text_raw: str):
        """Resolve the record's parent text to a canonical unit:
        synonym, then exact, then with a suffix added or removed."""
        if not parent_text_raw:
            return None
        norm = normalize(parent_text_raw, self.pool.data_dir)["normalized"]
        syn = self.pool.match_synonym(norm)
        if syn and syn["org_unit"]:
            return syn["org_unit"]
        exact = self.pool.match_exact(norm)
        if exact["status"] == "tek_eslesme":
            return exact["matches"][0]
        return self.flexible_exact_match(norm)

    def flexible_exact_match(self, normalized_text: str, candidate_uuids=None):
        """The canonical list itself isn't consistent: some entries carry the
        suffix ('MAKİNA MÜHENDİSLİĞİ BÖLÜMÜ'), some don't ('BİLGİSAYAR
        MÜHENDİSLİĞİ'). Try adding one, then try removing one. Only a single
        hit counts."""
        comp = self.try_suffix_completion(normalized_text, candidate_uuids)
        if comp["status"] == "tamamlandi":
            return comp["unit"]
        cls = structural_class(normalized_text)
        if cls:
            stripped = normalized_text[: -len(cls)].strip()
            exact = self.pool.match_exact(stripped, candidate_uuids)
            if exact["status"] == "tek_eslesme":
                return exact["matches"][0]
        return None

    def candidate_pool_for_parent(self, parent_unit: dict) -> list:
        """uuids of the parent itself and its whole subtree.

        The whole subtree, not just direct children: a record with parent
        "GENEL SEKRETERLİK" and text "PERSONEL DAİRE BAŞKANLIĞI" is an exact
        match, but that unit sits two levels down. With one level only, the
        pool had three units, nothing matched, and an exact hit went to review.

        The parent itself is included too. Some sources put the unit's own name
        in the "parent" field; for a leaf that gave an empty pool.
        """
        if not parent_unit:
            return []
        pool = [parent_unit["uuid"]]
        # BFS; the visited set guards against a cycle in bad data
        kuyruk = [parent_unit["uuid"]]
        ziyaret_edildi = {parent_unit["uuid"]}
        while kuyruk:
            simdiki = kuyruk.pop(0)
            for cocuk_uuid in self.pool.children_of.get(simdiki, []):
                if cocuk_uuid not in ziyaret_edildi:
                    ziyaret_edildi.add(cocuk_uuid)
                    pool.append(cocuk_uuid)
                    kuyruk.append(cocuk_uuid)
        return pool

    def try_suffix_completion(self, normalized_text: str, candidate_uuids=None):
        """'BİLGİSAYAR MÜHENDİSLİĞİ' -> 'BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ'.

        Accepted only if exactly one suffix produces a match; two or more is
        reported as ambiguous rather than guessed.
        """
        found = []
        for suf in STRUCTURAL_SUFFIXES:
            if normalized_text.endswith(suf):
                continue
            candidate_text = f"{normalized_text} {suf}"
            exact = self.pool.match_exact(candidate_text, candidate_uuids)
            if exact["status"] == "tek_eslesme":
                found.append((suf, exact["matches"][0]))
        if len(found) == 1:
            return {"status": "tamamlandi", "eklenen_ek": found[0][0], "unit": found[0][1]}
        if len(found) > 1:
            return {"status": "belirsiz", "adaylar": found}
        return {"status": "bulunamadi"}

    # step 6
    def constrained_fuzzy_search(self, normalized_text: str, candidate_uuids=None, top_k=5):
        """Top-k fuzzy candidates, limited to candidate_uuids when given.

        Searching the whole pool is allowed but riskier; the caller should
        lower confidence for it. Candidates that break a veto pair are never
        returned.
        """
        pool_units = (
            [self.pool.units[u] for u in candidate_uuids if u in self.pool.units]
            if candidate_uuids is not None
            else list(self.pool.units.values())
        )
        scored = []
        for unit in pool_units:
            cand_name = turkish_upper(unit["ad"].strip())
            if violates_veto(cand_name, normalized_text):
                continue
            score = word_similarity(cand_name, normalized_text)
            scored.append((score, unit))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


if __name__ == "__main__":
    pool = CanonicalPool()
    hz = HierarchyEngine(pool)

    print("--- veto ---")
    print("BİLGİ İŞLEM MÜDÜRLÜĞÜ vs BİLGİ İŞLEM DAİRE BAŞKANLIĞI ->",
          violates_veto("BİLGİ İŞLEM DAİRE BAŞKANLIĞI", "BİLGİ İŞLEM MÜDÜRLÜĞÜ"))
    print("BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ vs MÜHENDİSLİK FAKÜLTESİ ->",
          violates_veto("MÜHENDİSLİK FAKÜLTESİ", "BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ"))

    print()
    print("--- suffix completion ---")
    norm = normalize("Bilgisayar Mühendisliği")["normalized"]
    res = hz.try_suffix_completion(norm)
    print(f"{norm!r} ->", res["status"], res.get("eklenen_ek"),
          res["unit"]["ad"] if res.get("unit") else "")

    print()
    print("--- synonym, then missing suffix: MAKİNA MÜHENDİSLİĞİ ---")
    norm2 = normalize("Makine Mühendisliği")["normalized"]
    syn = pool.match_synonym(norm2)
    print("resmi_isim:", syn["resmi_isim"], "| org_unit found:", bool(syn["org_unit"]))
    resmi_norm = turkish_upper(syn["resmi_isim"].strip())
    res2 = hz.try_suffix_completion(resmi_norm)
    print(f"{resmi_norm!r} suffix completion ->", res2["status"], res2.get("eklenen_ek"),
          res2["unit"]["ad"] if res2.get("unit") else "")

    print()
    print("--- parent-limited search ---")
    parent = hz.resolve_parent_unit("Bilgisayar Mühendisliği Bölümü")
    print("parent:", parent["ad"] if parent else None)
    kids = hz.candidate_pool_for_parent(parent) if parent else []
    print("pool size:", len(kids))
