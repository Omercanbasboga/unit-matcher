"""Synonym and exact-name lookups against the canonical unit pool.

Steps 2 and 3 of the decision tree: if the normalised text is a known synonym,
or matches a canonical name exactly, we're done (confidence 0.95-1.00) and the
LLM is never called.
"""
import json
from collections import defaultdict
from pathlib import Path

from engine.normalize import normalize, turkish_upper
from engine.config import get_data_dir, DEFAULT_DATA_DIR


# Kept much shorter than db_sink's timeout because a pool is built at every
# script start and in every test. libpq tries each address of "localhost"
# (IPv6 then IPv4) with the full timeout, so 10s turned into ~20s of waiting
# when the DB was down. If Postgres is up it answers in milliseconds anyway.
_POOL_DB_CONNECT_TIMEOUT_SECONDS = 2


def _load_units_from_db(conn_params=None):
    """Read canonical_units from Postgres in the same shape as the JSON file.

    Returns None (never raises) if psycopg is missing, the DB is unreachable
    or the table is empty, so the caller can fall back to the file.
    """
    try:
        import psycopg
    except ImportError:
        return None
    from engine.db_sink import _connection_params
    params = conn_params or _connection_params()
    try:
        with psycopg.connect(**params, connect_timeout=_POOL_DB_CONNECT_TIMEOUT_SECONDS) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT uuid, ad, tur, kod, ust_uuid, ust_kod, node_id, kaynak "
                    "FROM canonical_units;"
                )
                rows = cur.fetchall()
    except Exception:
        return None
    if not rows:
        return None
    units = {}
    for uuid, ad, tur, kod, ust_uuid, ust_kod, node_id, kaynak in rows:
        uuid = str(uuid)
        units[uuid] = {
            "uuid": uuid, "ad": ad, "tur": tur, "kod": kod,
            "ust_uuid": str(ust_uuid) if ust_uuid else None, "ust_kod": ust_kod,
            "node_id": list(node_id or []), "kaynak": kaynak,
        }
    return units


def _load_units_from_file(data_dir):
    with open(data_dir / "canonical_units.json", encoding="utf-8") as f:
        return json.load(f)


def _load_units(data_dir, conn_params=None):
    """Postgres first, JSON file as fallback.

    feedback.approve_node_id() writes both on every approval, so the file is
    always a usable copy of the table. The DB is only consulted for the
    project's default data dir; any other data dir (another organisation,
    a test's temp folder) stays file-only so the table never leaks into it.
    """
    if conn_params is None:
        try:
            if Path(data_dir).resolve() != Path(DEFAULT_DATA_DIR).resolve():
                return _load_units_from_file(data_dir)
        except OSError:
            return _load_units_from_file(data_dir)
    db_units = _load_units_from_db(conn_params)
    if db_units:
        return db_units
    return _load_units_from_file(data_dir)


class CanonicalPool:
    """Loads and indexes canonical units, synonyms and staffing rules.

    Nothing here is tied to one organisation; it only expects the JSON shapes
    in data/.
    """

    def __init__(self, data_dir=None, db_conn_params=None):
        data_dir = data_dir or get_data_dir()
        self.data_dir = data_dir
        self.units = _load_units(data_dir, conn_params=db_conn_params)  # uuid -> unit dict
        with open(data_dir / "synonyms.json", encoding="utf-8") as f:
            self.synonyms = json.load(f)  # ALT_ISIM(upper) -> {resmi_isim, kanonik_id}
        with open(data_dir / "kadro_kurallari.json", encoding="utf-8") as f:
            self.kadro_kurallari = json.load(f)

        # Matches promoted from human review (see engine/feedback.py).
        # Optional: a data dir without these files just starts empty.
        self.learned_corrections = self._load_optional_json(data_dir / "learned_corrections.json")
        self.learned_id_aliases = self._load_optional_json(data_dir / "learned_id_aliases.json")

        # Optional snapshot of cross-source ID conflicts (same raw ID resolved
        # to different units in different sources, or one unit arriving under
        # several raw IDs). Exported by a separate job; the engine runs fine
        # without it.
        capraz_kaynak = self._load_optional_json(data_dir / "capraz_kaynak_id_uyarilari.json")
        self._id_celiskileri = capraz_kaynak.get("id_celiskileri", {})
        self._birim_coklu_id = capraz_kaynak.get("birim_coklu_id", {})
        # Stored as corrected_id -> group; lookups go the other way, so build
        # a reverse index.
        self._incoming_id_to_coklu_birim = {}
        for corrected_id, grup in self._birim_coklu_id.items():
            for satir in grup.get("gelen_idler", []):
                self._incoming_id_to_coklu_birim[str(satir.get("incoming_id"))] = {
                    "corrected_id": corrected_id,
                    "corrected_text": grup.get("corrected_text"),
                }

        self.by_kod = {}
        self.by_alias_id = {}
        self.by_name = defaultdict(list)
        self.children_of = defaultdict(list)  # ust_uuid -> [uuid, ...]
        for uuid, unit in self.units.items():
            self.by_kod[unit["kod"]] = unit
            self.by_name[turkish_upper(unit["ad"].strip())].append(unit)
            if unit.get("ust_uuid"):
                self.children_of[unit["ust_uuid"]].append(uuid)

        # by_alias_id: extra IDs a unit is known by in other systems (the
        # "node_id" list, either declared by the source export or added by
        # feedback.approve_node_id() after a human approves it).
        # An ID claimed by two different units is ambiguous and left out of
        # the index entirely; approve_node_id() removes an ID from every other
        # unit before adding it, so approvals never hit this case.
        # An alias that equals another unit's official code is dropped too: the
        # official code is the stronger evidence. Otherwise resolve_by_id(),
        # which has no text to cross-check, could map that ID to the wrong
        # unit at 0.97 without anyone noticing.
        alias_raw = defaultdict(set)
        for uuid, unit in self.units.items():
            for alias in unit.get("node_id", []):
                alias_raw[alias].add(uuid)
        for alias, uuids in alias_raw.items():
            if alias in self.by_kod:
                continue
            if len(uuids) == 1:
                self.by_alias_id[alias] = self.units[next(iter(uuids))]

    @staticmethod
    def _load_optional_json(path):
        if not path.is_file():
            return {}
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        return json.loads(content) if content else {}

    # step 2.5: learned from human review
    def match_learned(self, normalized_text: str):
        """Text match learned from reviewers (learned_corrections.json).

        Confidence is deliberately below the synonym table (0.90-0.93 vs 0.97):
        it comes from one or two reviewers, not from the official list. It
        creeps up by 0.01 per extra approval.
        """
        entry = self.learned_corrections.get(normalized_text)
        if not entry:
            return None
        org_unit = self.units.get(entry.get("kanonik_uuid"))
        if org_unit is None:
            return None
        onay_sayisi = entry.get("onay_sayisi", 1)
        confidence = min(0.90 + 0.01 * (onay_sayisi - 1), 0.93)
        return {
            "resmi_isim": org_unit["ad"],
            "org_unit": org_unit,
            "confidence": confidence,
            "reason": (
                f"Insan onayi ile ogrenilmis eslesme (HITL geri besleme "
                f"dongusu, onay_sayisi={onay_sayisi}) - komisyon-onayli "
                "senonim tablosundan degil."
            ),
            "layer": "llm",
        }

    def is_known_incoming_id(self, incoming_id) -> bool:
        """Is this raw ID already known, either as a canonical uuid/code or as
        an approved node_id alias?

        Known IDs skip the cross-source suspicion check: once an ID is known or
        a person has approved it, the engine shouldn't second-guess it.
        """
        if incoming_id is None:
            return False
        incoming_id = str(incoming_id)
        return incoming_id in self.units or incoming_id in self.by_alias_id

    def flagged_incoming_id_warning(self, incoming_id) -> str:
        """Explain why a raw ID is suspicious across sources, or return None.

        This only produces the explanation for the reviewer; the decision to
        skip automatic/LLM resolution is made by the caller.
        """
        if incoming_id is None:
            return None
        incoming_id = str(incoming_id)

        celiski = self._id_celiskileri.get(incoming_id)
        if celiski:
            detay = "; ".join(
                f"{s['kaynak']}->{s['corrected_text']}({s['corrected_id']})" for s in celiski
            )
            return (
                f"Capraz Kaynak Kontrolu: '{incoming_id}' ID'si farkli kaynaklarda "
                f"farkli kanonik birime cozulmus ({detay}) - bu ID'ye guvenerek "
                "otomatik/LLM karari verilmedi, insan onayina birakildi."
            )

        coklu = self._incoming_id_to_coklu_birim.get(incoming_id)
        if coklu:
            return (
                f"Capraz Kaynak Kontrolu: '{incoming_id}' ID'si, '{coklu['corrected_text']}' "
                f"({coklu['corrected_id']}) biriminin kaynaklar arasinda bilinen BIRDEN FAZLA "
                "ham ID'sinden biri - bu birim baska bir ID ile de geliyor olabilir, "
                "insan onayina birakildi (crosswalk'a eklenmeye aday olabilir)."
            )
        return None

    def match_learned_id(self, incoming_id):
        """ID-based counterpart of match_learned (learned_id_aliases.json)."""
        entry = self.learned_id_aliases.get(str(incoming_id))
        if not entry:
            return None
        org_unit = self.units.get(entry.get("kanonik_uuid"))
        if org_unit is None:
            return None
        onay_sayisi = entry.get("onay_sayisi", 1)
        confidence = min(0.90 + 0.01 * (onay_sayisi - 1), 0.93)
        return {"org_unit": org_unit, "confidence": confidence, "onay_sayisi": onay_sayisi}

    # step 2: synonym table
    def match_synonym(self, normalized_text: str):
        """Look the normalised text up in the synonym table.

        The ID in the synonym table is not always the unit's org code. For
        "BAP KOORDİNATÖRLÜĞÜ" the table says 1541, but the unit's code in the
        org tree is 180; 1541 is actually its staffing-type ID from the kadro
        rules. So both are returned and the orchestrator decides which to use:
          org_unit            - the real unit, found by its official name
          senonim_tablosu_id  - the raw ID from the synonym table
        """
        entry = self.synonyms.get(normalized_text)
        if not entry:
            return None

        resmi_isim = entry["resmi_isim"]
        senonim_id = entry["kanonik_id"]

        org_unit = None
        exact = self.match_exact(turkish_upper(resmi_isim.strip()))
        if exact["status"] == "tek_eslesme":
            org_unit = exact["matches"][0]

        id_uyusmazligi = bool(org_unit) and org_unit["kod"] != senonim_id

        return {
            "resmi_isim": resmi_isim,
            "org_unit": org_unit,
            "senonim_tablosu_id": senonim_id,
            "id_uyusmazligi": id_uyusmazligi,
            "confidence": 0.97,
            "reason": (
                "Onaylanmis senonim tablosunda birebir eslesme bulundu."
                + (
                    f" UYARI: senonim tablosu ID'si ({senonim_id}) org-hiyerarsi "
                    f"kodundan ({org_unit['kod'] if org_unit else 'YOK'}) farkli; "
                    "kadro_kurallari baglaminda kullanilmasi gerekebilir."
                    if id_uyusmazligi else ""
                )
            ),
            "layer": "llm",
        }

    # step 3: exact name match
    def match_exact(self, normalized_text: str, candidate_uuids: list = None):
        """Exact lookup by normalised name, optionally limited to candidate_uuids
        (the subtree picked in step 4).

        Returns {"status": "tek_eslesme" | "belirsiz" | "bulunamadi", "matches": [...]}.
        """
        matches = self.by_name.get(normalized_text, [])
        if candidate_uuids is not None:
            allowed = set(candidate_uuids)
            matches = [m for m in matches if m["uuid"] in allowed]

        if len(matches) == 0:
            return {"status": "bulunamadi", "matches": []}
        if len(matches) == 1:
            return {"status": "tek_eslesme", "matches": matches}
        return {"status": "belirsiz", "matches": matches}

    def parent_chain(self, uuid: str) -> list:
        """Ancestors of a unit, nearest first."""
        chain = []
        seen = set()
        current = self.units.get(uuid)
        while current and current.get("ust_uuid") and current["ust_uuid"] not in seen:
            parent = self.units.get(current["ust_uuid"])
            if not parent:
                break
            chain.append(parent)
            seen.add(parent["uuid"])
            current = parent
        return chain


if __name__ == "__main__":
    pool = CanonicalPool()
    print(f"{len(pool.units)} kanonik birim, {len(pool.synonyms)} senonim yuklendi.")

    tests = [
        "BAP KOORDİNATÖRLÜĞÜ",           # synonym
        "MAKİNE MÜHENDİSLİĞİ",            # synonym
        "EĞİTİM FAKÜLTESİ",               # exact
        "BEDEN EĞİTİMİ BÖLÜMÜ",           # ambiguous, two units share the name
        "OLMAYAN BİR BİRİM",              # not found
    ]
    for t in tests:
        norm = normalize(t)["normalized"]
        syn = pool.match_synonym(norm)
        if syn:
            org_kod = syn["org_unit"]["kod"] if syn["org_unit"] else None
            print(f"[SENONIM] {t!r} -> {syn['resmi_isim']} (org_kod={org_kod}, "
                  f"senonim_tablosu_id={syn['senonim_tablosu_id']}, uyusmazlik={syn['id_uyusmazligi']})")
            continue
        exact = pool.match_exact(norm)
        if exact["status"] == "tek_eslesme":
            m = exact["matches"][0]
            print(f"[TAM ESLESME] {t!r} -> {m['ad']} (kod={m['kod']})")
        elif exact["status"] == "belirsiz":
            print(f"[BELIRSIZ] {t!r} -> {len(exact['matches'])} aday: " +
                  ", ".join(f"kod={m['kod']}/ust_kod={m.get('ust_kod')}" for m in exact["matches"]))
        else:
            print(f"[BULUNAMADI] {t!r}")
