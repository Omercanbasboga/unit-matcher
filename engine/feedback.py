"""Human review feedback: record decisions and learn from them.

Fields the engine couldn't resolve go to a person. Their answers are kept
here and, once they agree, promoted into lookup tables so the same dirty
text or ID doesn't come back to review on the next run.

Channels:
  learned_corrections.json  text -> unit, from resolve()
  learned_id_aliases.json   raw ID -> unit, from resolve_by_id() (older channel)
  approve_node_id()         one approval is enough: writes the ID straight into
                            the unit's node_id list (file and Postgres), with the
                            same weight as an ID the source system declared
  kanonik_eksik_bildirimleri.json
                            "the right unit isn't in the canonical list at all";
                            a data gap, not a match, so it never feeds the tables

bulk_approve_node_ids() is what the run script uses for IDs the engine learned
on its own during a run.

Everything is plain JSON files. approve_node_id() also updates Postgres when
psycopg and a DB are available; if not, the file side still succeeds and the
returned "db" entry says what failed.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from engine.config import get_data_dir
from engine.vector_store import VectorStore

HUMAN_DECISIONS_LOG = "human_decisions_log.json"
LEARNED_CORRECTIONS = "learned_corrections.json"
LEARNED_ID_ALIASES = "learned_id_aliases.json"
MISSING_CANONICAL_REPORTS = "kanonik_eksik_bildirimleri.json"

KARAR_TIPI_METIN = "metin_eslesme"
KARAR_TIPI_ID = "id_eslesme"


def default_reviewer() -> str:
    """OS user name, used when no reviewer is given."""
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return os.environ.get("USERNAME") or os.environ.get("USER") or "bilinmiyor"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default):
    if not path.is_file():
        return default
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return default
    return json.loads(content)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_decision(
    data_dir=None, *, record_id, field_name, karar_tipi,
    secilen_uuid, secilen_ad=None, secilen_kod=None,
    incoming_text=None, incoming_id=None, normalized_text=None,
    onceki_confidence=None, onceki_reason=None, reviewer=None, not_=None,
) -> dict:
    """Append one decision to human_decisions_log.json.

    karar_tipi is KARAR_TIPI_METIN (text-based) or KARAR_TIPI_ID (ID only).
    This doesn't validate secilen_uuid; the caller should check it against
    the CanonicalPool first.
    """
    data_dir = Path(data_dir) if data_dir else get_data_dir()
    path = data_dir / HUMAN_DECISIONS_LOG
    log = _read_json(path, [])

    entry = {
        "timestamp": _now_iso(),
        "reviewer": reviewer or default_reviewer(),
        "record_id": record_id,
        "field_name": field_name,
        "karar_tipi": karar_tipi,
        "incoming_text": incoming_text,
        "incoming_id": incoming_id,
        "normalized_text": normalized_text,
        "onceki_confidence": onceki_confidence,
        "onceki_reason": onceki_reason,
        "secilen_uuid": secilen_uuid,
        "secilen_ad": secilen_ad,
        "secilen_kod": secilen_kod,
        "not": not_,
    }
    log.append(entry)
    _write_json(path, log)
    return entry


def report_missing_canonical(
    data_dir=None, *, record_id, field_name,
    incoming_text=None, incoming_id=None, normalized_text=None,
    onceki_reason=None, reviewer=None, not_=None,
) -> dict:
    """The reviewer says the right unit isn't in the canonical list.
    Goes to a separate backlog, to be handled in the next data update."""
    data_dir = Path(data_dir) if data_dir else get_data_dir()
    path = data_dir / MISSING_CANONICAL_REPORTS
    reports = _read_json(path, [])

    entry = {
        "timestamp": _now_iso(),
        "reviewer": reviewer or default_reviewer(),
        "record_id": record_id,
        "field_name": field_name,
        "incoming_text": incoming_text,
        "incoming_id": incoming_id,
        "normalized_text": normalized_text,
        "onceki_reason": onceki_reason,
        "not": not_,
    }
    reports.append(entry)
    _write_json(path, reports)
    return entry


def approve_node_id(
    data_dir=None, *, node_id, kanonik_uuid, secilen_ad=None, secilen_kod=None,
    record_id=None, field_name=None, reviewer=None, not_=None, conn_params=None,
) -> dict:
    """A reviewer says "this raw ID belongs to that unit". Applied at once.

    The ID is first removed from any other unit, then added to this one, so
    an ID always belongs to exactly one unit (by_alias_id drops IDs claimed
    by two). Writes canonical_units.json and the Postgres table; the file is
    updated even if the DB isn't reachable. Also logged via record_decision.

    Returns {"basarili", "sebep" (on failure), "db"}.
    """
    data_dir = Path(data_dir) if data_dir else get_data_dir()
    node_id = str(node_id)

    units_path = data_dir / "canonical_units.json"
    units = _read_json(units_path, {})
    target = units.get(kanonik_uuid)
    if target is None:
        return {
            "basarili": False,
            "sebep": f"kanonik_uuid '{kanonik_uuid}' data_dir/canonical_units.json icinde bulunamadi.",
        }

    record_decision(
        data_dir, record_id=record_id or f"node-id-onayi-{node_id}",
        field_name=field_name or "node_id_onayi", karar_tipi=KARAR_TIPI_ID,
        secilen_uuid=kanonik_uuid, secilen_ad=secilen_ad or target.get("ad"),
        secilen_kod=secilen_kod or target.get("kod"),
        incoming_id=node_id, reviewer=reviewer, not_=not_,
    )

    for uuid, unit in units.items():
        if uuid != kanonik_uuid and node_id in unit.get("node_id", []):
            unit["node_id"].remove(node_id)
    node_list = target.setdefault("node_id", [])
    if node_id not in node_list:
        node_list.append(node_id)
    _write_json(units_path, units)

    from engine.db_sink import update_canonical_unit_node_id
    db_sonuc = update_canonical_unit_node_id(kanonik_uuid, node_id, conn_params=conn_params)

    return {"basarili": True, "db": db_sonuc}


def bulk_approve_node_ids(entries: list, data_dir=None, conn_params=None, reviewer=None) -> dict:
    """approve_node_id() for many IDs at once.

    Used for IDs the engine learned during a run. Doing them one by one
    meant re-reading and re-writing the JSON and opening a DB connection per
    ID; this reads and writes the file once and uses one connection.

    entries has the shape of MatchEngine.auto_learned_node_ids. They're logged
    with reviewer "otomatik-motor" so they never get mixed up with a real
    person's decisions.

    Returns {"basarili", "ogrenilen_kayit", "db"}.
    """
    if not entries:
        return {"basarili": True, "ogrenilen_kayit": 0}

    data_dir = Path(data_dir) if data_dir else get_data_dir()
    reviewer = reviewer or "otomatik-motor"

    units_path = data_dir / "canonical_units.json"
    units = _read_json(units_path, {})

    uygulanan = []
    for entry in entries:
        kanonik_uuid, node_id = entry["kanonik_uuid"], str(entry["node_id"])
        target = units.get(kanonik_uuid)
        if target is None:
            continue  # file and in-memory pool out of sync: skip this one, keep going
        for uuid, unit in units.items():
            if uuid != kanonik_uuid and node_id in unit.get("node_id", []):
                unit["node_id"].remove(node_id)
        node_list = target.setdefault("node_id", [])
        if node_id not in node_list:
            node_list.append(node_id)
        uygulanan.append(entry)

        record_decision(
            data_dir, record_id=entry.get("record_id") or f"node-id-otomatik-{node_id}",
            field_name=entry.get("field_name") or "node_id_otomatik_ogrenme",
            karar_tipi=KARAR_TIPI_ID,
            secilen_uuid=kanonik_uuid,
            secilen_ad=entry.get("secilen_ad") or target.get("ad"),
            secilen_kod=entry.get("secilen_kod") or target.get("kod"),
            incoming_id=node_id, reviewer=reviewer, not_=entry.get("not_"),
        )

    _write_json(units_path, units)

    from engine.db_sink import update_canonical_unit_node_ids_bulk
    db_sonuc = update_canonical_unit_node_ids_bulk(
        [{"kanonik_uuid": e["kanonik_uuid"], "node_id": e["node_id"]} for e in uygulanan],
        conn_params=conn_params,
    )

    return {"basarili": True, "ogrenilen_kayit": len(uygulanan), "db": db_sonuc}


def promote_to_learned_corrections(data_dir=None, min_agreement: int = 1, vector_store: VectorStore = None) -> dict:
    """Group logged decisions by text / ID and promote the ones that agree.

    A key is promoted when every decision for it picked the same unit and
    there are at least min_agreement of them. Keys with conflicting
    decisions are not promoted, and if one was promoted earlier it's taken
    back out: three reviewers say A, a fourth says B, and A should stop
    being applied automatically until someone sorts it out.

    Promoted text matches are also added to the vector store.

    Returns {"terfi_edilen", "celisen", "yetersiz_onay"}.
    """
    data_dir = Path(data_dir) if data_dir else get_data_dir()
    if vector_store is None:
        vector_store = VectorStore(data_dir=data_dir)
    log = _read_json(data_dir / HUMAN_DECISIONS_LOG, [])

    metin_gruplari, id_gruplari = {}, {}
    for entry in log:
        if entry.get("secilen_uuid") is None:
            continue
        if entry["karar_tipi"] == KARAR_TIPI_METIN and entry.get("normalized_text"):
            metin_gruplari.setdefault(entry["normalized_text"], []).append(entry)
        elif entry["karar_tipi"] == KARAR_TIPI_ID and entry.get("incoming_id"):
            id_gruplari.setdefault(str(entry["incoming_id"]), []).append(entry)

    terfi_edilen, celisen, yetersiz_onay = [], [], []

    def _process(gruplar, out_path, tur):
        store = _read_json(out_path, {})
        for anahtar, entries in gruplar.items():
            uuids = {e["secilen_uuid"] for e in entries}
            if len(uuids) > 1:
                geri_alindi = anahtar in store
                if geri_alindi:
                    del store[anahtar]
                celisen.append({
                    "anahtar": anahtar, "tur": tur, "adaylar": sorted(uuids),
                    "geri_alindi": geri_alindi,
                })
                continue
            if len(entries) < min_agreement:
                yetersiz_onay.append({"anahtar": anahtar, "tur": tur, "onay_sayisi": len(entries)})
                continue
            last = entries[-1]
            store[anahtar] = {
                "resmi_isim": last.get("secilen_ad"),
                "kanonik_id": last.get("secilen_kod"),
                "kanonik_uuid": last["secilen_uuid"],
                "kaynak": "insan_onayi",
                "onay_sayisi": len(entries),
                "son_karar_tarihi": last["timestamp"],
            }
            terfi_edilen.append({"anahtar": anahtar, "tur": tur,
                                  "secilen_uuid": last["secilen_uuid"], "onay_sayisi": len(entries)})
            if tur == KARAR_TIPI_METIN:
                vector_store.upsert(
                    anahtar,
                    org_unit={"uuid": last["secilen_uuid"], "ad": last.get("secilen_ad"),
                              "kod": last.get("secilen_kod")},
                    layer="insan", confidence=1.0, kaynak="hitl_promote",
                )
        _write_json(out_path, store)

    _process(metin_gruplari, data_dir / LEARNED_CORRECTIONS, KARAR_TIPI_METIN)
    _process(id_gruplari, data_dir / LEARNED_ID_ALIASES, KARAR_TIPI_ID)

    return {"terfi_edilen": terfi_edilen, "celisen": celisen, "yetersiz_onay": yetersiz_onay}
