"""Apply db/schema.sql and load data/*.json into Postgres.

Loads the reference tables and the review/RAG tables as they are (empty files
give empty tables), the hierarchy paths, and any cikti/*_eslesen.json results.

human_decisions_log is skipped unless --with-human-decisions-log is given.
Loading it truncates the table first. Once reviews are written to Postgres
directly, the JSON file is no longer the newer copy, and a routine re-seed
would wipe real decisions.

  python scripts/seed_postgres.py
  python scripts/seed_postgres.py --skip-eslesen
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import get_data_dir  # noqa: E402
from engine.db_sink import write_matched_records, _connection_params  # noqa: E402
from scripts._env import load_dotenv  # noqa: E402


def run_schema(conn) -> None:
    schema_sql = (PROJECT_ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()
    print("[seed_postgres] db/schema.sql uygulandi (idempotent - IF NOT EXISTS).")


def seed_canonical_units(conn, data_dir: Path) -> None:
    with open(data_dir / "canonical_units.json", encoding="utf-8") as f:
        units = json.load(f)

    rows = [
        (
            u["uuid"], u["ad"], u.get("tur"), u.get("kod"),
            u.get("ust_uuid"), u.get("ust_kod"),
            u.get("node_id", []), u.get("kaynak"),
        )
        for u in units.values()
    ]
    with conn.cursor() as cur:
        # the ust_uuid FK is deferred, so rows can go in any order
        cur.executemany(
            """
            INSERT INTO canonical_units (uuid, ad, tur, kod, ust_uuid, ust_kod, node_id, kaynak)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (uuid) DO UPDATE SET
                ad = EXCLUDED.ad, tur = EXCLUDED.tur, kod = EXCLUDED.kod,
                ust_uuid = EXCLUDED.ust_uuid, ust_kod = EXCLUDED.ust_kod,
                node_id = EXCLUDED.node_id, kaynak = EXCLUDED.kaynak;
            """,
            rows,
        )
    conn.commit()
    print(f"[seed_postgres] canonical_units: {len(rows)} satir yuklendi.")


def seed_synonyms(conn, data_dir: Path) -> None:
    with open(data_dir / "synonyms.json", encoding="utf-8") as f:
        syns = json.load(f)
    rows = [(alt, entry["resmi_isim"], entry.get("kanonik_id")) for alt, entry in syns.items()]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO synonyms (alt_isim, resmi_isim, kanonik_id) VALUES (%s, %s, %s)
            ON CONFLICT (alt_isim) DO UPDATE SET
                resmi_isim = EXCLUDED.resmi_isim, kanonik_id = EXCLUDED.kanonik_id;
            """,
            rows,
        )
    conn.commit()
    print(f"[seed_postgres] synonyms: {len(rows)} satir yuklendi.")


def seed_abbreviations(conn, data_dir: Path) -> None:
    with open(data_dir / "abbreviations.json", encoding="utf-8") as f:
        abbr = json.load(f)
    rows = list(abbr.items())
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO abbreviations (kisaltma, resmi_ad) VALUES (%s, %s)
            ON CONFLICT (kisaltma) DO UPDATE SET resmi_ad = EXCLUDED.resmi_ad;
            """,
            rows,
        )
    conn.commit()
    print(f"[seed_postgres] abbreviations: {len(rows)} satir yuklendi.")


def seed_kadro_kurallari(conn, data_dir: Path) -> None:
    with open(data_dir / "kadro_kurallari.json", encoding="utf-8") as f:
        rows_raw = json.load(f)
    rows = [
        (r["birim_adi"], r.get("idari_id"), r.get("diger_personel_id"), r.get("akademik_id"))
        for r in rows_raw
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO kadro_kurallari (birim_adi, idari_id, diger_personel_id, akademik_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (birim_adi) DO UPDATE SET
                idari_id = EXCLUDED.idari_id, diger_personel_id = EXCLUDED.diger_personel_id,
                akademik_id = EXCLUDED.akademik_id;
            """,
            rows,
        )
    conn.commit()
    print(f"[seed_postgres] kadro_kurallari: {len(rows)} satir yuklendi.")


def _load_json_or_default(data_dir: Path, filename: str, default):
    path = data_dir / filename
    if not path.is_file():
        print(f"[seed_postgres] UYARI - {filename} yok, atlandi.")
        return default
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    return json.loads(content) if content else default


def seed_human_decisions_log(conn, data_dir: Path) -> None:
    """Truncate and reload the decision log from JSON. Only with
    --with-human-decisions-log (see the module docstring)."""
    entries = _load_json_or_default(data_dir, "human_decisions_log.json", [])
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE human_decisions_log;")
        if entries:
            cur.executemany(
                """
                INSERT INTO human_decisions_log
                    ("timestamp", reviewer, record_id, field_name, karar_tipi,
                     incoming_text, incoming_id, normalized_text,
                     onceki_confidence, onceki_reason,
                     secilen_uuid, secilen_ad, secilen_kod, not_)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                [
                    (
                        e["timestamp"], e["reviewer"], e["record_id"], e["field_name"], e["karar_tipi"],
                        e.get("incoming_text"), e.get("incoming_id"), e.get("normalized_text"),
                        e.get("onceki_confidence"), e.get("onceki_reason"),
                        e.get("secilen_uuid"), e.get("secilen_ad"), e.get("secilen_kod"), e.get("not"),
                    )
                    for e in entries
                ],
            )
    conn.commit()
    print(f"[seed_postgres] human_decisions_log: {len(entries)} satir yuklendi.")


def seed_kanonik_eksik_bildirimleri(conn, data_dir: Path) -> None:
    """No natural key, so truncate and reload."""
    entries = _load_json_or_default(data_dir, "kanonik_eksik_bildirimleri.json", [])
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE kanonik_eksik_bildirimleri;")
        if entries:
            cur.executemany(
                """
                INSERT INTO kanonik_eksik_bildirimleri
                    ("timestamp", reviewer, record_id, field_name,
                     incoming_text, incoming_id, normalized_text, onceki_reason, not_,
                     kapatildi_mi, kapatan, kapatma_tarihi)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                [
                    (
                        e["timestamp"], e["reviewer"], e["record_id"], e["field_name"],
                        e.get("incoming_text"), e.get("incoming_id"), e.get("normalized_text"),
                        e.get("onceki_reason"), e.get("not"),
                        e.get("kapatildi_mi", False), e.get("kapatan"), e.get("kapatma_tarihi"),
                    )
                    for e in entries
                ],
            )
    conn.commit()
    print(f"[seed_postgres] kanonik_eksik_bildirimleri: {len(entries)} satir yuklendi.")


def seed_learned_corrections(conn, data_dir: Path) -> None:
    """Upsert, keyed by normalized_text."""
    entries = _load_json_or_default(data_dir, "learned_corrections.json", {})
    rows = [
        (
            normalized_text, e.get("resmi_isim"), e.get("kanonik_id"), e.get("kanonik_uuid"),
            e.get("kaynak"), e.get("onay_sayisi", 1), e.get("son_karar_tarihi"),
        )
        for normalized_text, e in entries.items()
    ]
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO learned_corrections
                    (normalized_text, resmi_isim, kanonik_id, kanonik_uuid, kaynak, onay_sayisi, son_karar_tarihi)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (normalized_text) DO UPDATE SET
                    resmi_isim = EXCLUDED.resmi_isim, kanonik_id = EXCLUDED.kanonik_id,
                    kanonik_uuid = EXCLUDED.kanonik_uuid, kaynak = EXCLUDED.kaynak,
                    onay_sayisi = EXCLUDED.onay_sayisi, son_karar_tarihi = EXCLUDED.son_karar_tarihi;
                """,
                rows,
            )
        conn.commit()
    print(f"[seed_postgres] learned_corrections: {len(rows)} satir yuklendi.")


def seed_learned_id_aliases(conn, data_dir: Path) -> None:
    """Upsert, keyed by incoming_id."""
    entries = _load_json_or_default(data_dir, "learned_id_aliases.json", {})
    rows = [
        (
            incoming_id, e.get("resmi_isim"), e.get("kanonik_id"), e.get("kanonik_uuid"),
            e.get("kaynak"), e.get("onay_sayisi", 1), e.get("son_karar_tarihi"),
        )
        for incoming_id, e in entries.items()
    ]
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO learned_id_aliases
                    (incoming_id, resmi_isim, kanonik_id, kanonik_uuid, kaynak, onay_sayisi, son_karar_tarihi)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (incoming_id) DO UPDATE SET
                    resmi_isim = EXCLUDED.resmi_isim, kanonik_id = EXCLUDED.kanonik_id,
                    kanonik_uuid = EXCLUDED.kanonik_uuid, kaynak = EXCLUDED.kaynak,
                    onay_sayisi = EXCLUDED.onay_sayisi, son_karar_tarihi = EXCLUDED.son_karar_tarihi;
                """,
                rows,
            )
        conn.commit()
    print(f"[seed_postgres] learned_id_aliases: {len(rows)} satir yuklendi.")


def seed_vector_index(conn, data_dir: Path) -> None:
    """vector_index.json is created by local runs (and git-ignored); skipped if absent."""
    entries = _load_json_or_default(data_dir, "vector_index.json", {})
    rows = [
        (
            normalized_text, e.get("kanonik_uuid"), e.get("resmi_isim"), e.get("kanonik_kod"),
            e.get("layer"), e.get("confidence"), e.get("kaynak"), e.get("record_id"),
        )
        for normalized_text, e in entries.items()
    ]
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO vector_index
                    (normalized_text, kanonik_uuid, resmi_isim, kanonik_kod, layer, confidence, kaynak, record_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (normalized_text) DO UPDATE SET
                    kanonik_uuid = EXCLUDED.kanonik_uuid, resmi_isim = EXCLUDED.resmi_isim,
                    kanonik_kod = EXCLUDED.kanonik_kod, layer = EXCLUDED.layer,
                    confidence = EXCLUDED.confidence, kaynak = EXCLUDED.kaynak, record_id = EXCLUDED.record_id;
                """,
                rows,
            )
        conn.commit()
    print(f"[seed_postgres] vector_index: {len(rows)} satir yuklendi.")


def seed_birim_hiyerarsi(conn_params: dict, data_dir: Path) -> None:
    """Write the full root path of every unit (derived from canonical_units)."""
    from engine.canonical_lookup import CanonicalPool
    from engine.schema import full_hierarchy_chain
    from engine.db_sink import write_birim_hiyerarsi

    pool = CanonicalPool(data_dir=data_dir)
    rows = []
    for unit in pool.units.values():
        chain = full_hierarchy_chain(pool, unit)
        rows.append({
            "uuid": unit["uuid"], "ad": unit["ad"], "kod": unit.get("kod"),
            "zincir": chain["zincir"], "zincir_metni": chain["zincir_metni"],
        })

    sonuc = write_birim_hiyerarsi(rows, conn_params=conn_params)
    if sonuc["basarili"]:
        print(f"[seed_postgres] birim_hiyerarsi: {sonuc['yazilan_kayit']} satir yuklendi.")
    else:
        print(f"[seed_postgres] HATA - birim_hiyerarsi yazilamadi: {sonuc['sebep']}")


def seed_eslesen_kayitlar(conn_params: dict) -> None:
    """Load every cikti/<kaynak>_eslesen.json through db_sink.write_matched_records."""
    cikti_dir = PROJECT_ROOT / "cikti"
    paths = sorted(cikti_dir.glob("*_eslesen.json")) if cikti_dir.is_dir() else []
    if not paths:
        print("[seed_postgres] no cikti/*_eslesen.json, skipped.")
        return
    for path in paths:
        kaynak = path.name[: -len("_eslesen.json")]
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        sonuc = write_matched_records(kaynak, records, conn_params=conn_params)
        if sonuc["basarili"]:
            print(f"[seed_postgres] eslesen_kayitlar <- {kaynak}: {sonuc['yazilan_kayit']} satir yazildi.")
        else:
            print(f"[seed_postgres] HATA - {kaynak} yazilamadi: {sonuc['sebep']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-eslesen", action="store_true",
                     help="cikti/*_eslesen.json yukleme adimini atla (sadece referans veri).")
    ap.add_argument("--with-human-decisions-log", action="store_true",
                     help="Truncate human_decisions_log and reload it from JSON. Destroys decisions "
                          "that only exist in Postgres; for first setup or a deliberate restore.")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    import psycopg
    conn_params = _connection_params()
    print(f"[seed_postgres] Baglaniliyor: {conn_params['host']}:{conn_params['port']}/{conn_params['dbname']} "
          f"(kullanici={conn_params['user']})")

    with psycopg.connect(**conn_params) as conn:
        run_schema(conn)

        data_dir = get_data_dir()
        seed_canonical_units(conn, data_dir)
        seed_synonyms(conn, data_dir)
        seed_abbreviations(conn, data_dir)
        seed_kadro_kurallari(conn, data_dir)
        if args.with_human_decisions_log:
            seed_human_decisions_log(conn, data_dir)
        else:
            print("[seed_postgres] human_decisions_log skipped (use --with-human-decisions-log).")
        seed_kanonik_eksik_bildirimleri(conn, data_dir)
        seed_learned_corrections(conn, data_dir)
        seed_learned_id_aliases(conn, data_dir)
        seed_vector_index(conn, data_dir)

    seed_birim_hiyerarsi(conn_params, data_dir)

    if not args.skip_eslesen:
        seed_eslesen_kayitlar(conn_params)

    print("[seed_postgres] Tamamlandi.")


if __name__ == "__main__":
    main()
