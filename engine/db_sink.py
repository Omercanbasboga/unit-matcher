"""Optional PostgreSQL output.

Writes resolved records, the review queue, approved node_ids and the unit
hierarchy. psycopg is imported lazily and nothing in here ever raises: if the
package is missing or the server is down, the call returns
{"basarili": False, "sebep": ...} and the file outputs carry on as normal.

Connection settings come from the standard PG* environment variables.
"""
import json
import os

# libpq's default against an unreachable host can be ~2 minutes (OS TCP
# retries). The run scripts try the DB on every run, so cap it.
CONNECT_TIMEOUT_SECONDS = 10

TABLE_NAME = "eslesen_kayitlar"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    kaynak TEXT NOT NULL,
    record_id TEXT NOT NULL,
    dominant_branch TEXT,
    overall_confidence NUMERIC(4,2),
    fields JSONB NOT NULL,
    guncellenme_zamani TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kaynak, record_id)
);
"""

_CREATE_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_fields ON {TABLE_NAME} USING GIN (fields);
"""

_UPSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (kaynak, record_id, dominant_branch, overall_confidence, fields, guncellenme_zamani)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (kaynak, record_id) DO UPDATE SET
    dominant_branch = EXCLUDED.dominant_branch,
    overall_confidence = EXCLUDED.overall_confidence,
    fields = EXCLUDED.fields,
    guncellenme_zamani = now();
"""


def _connection_params() -> dict:
    """PG* variables with local defaults."""
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": os.environ.get("PGPORT", "5432"),
        "dbname": os.environ.get("PGDATABASE", "unit_matcher"),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", ""),
    }


def _row_from_record(kaynak: str, record: dict) -> tuple:
    """Record dict -> upsert tuple. No DB access, so it can be tested alone."""
    return (
        kaynak,
        record["record_id"],
        record.get("dominant_branch"),
        record.get("overall_confidence"),
        json.dumps(record.get("fields", []), ensure_ascii=False),
    )


def _write_rows(create_sql: str, index_sql: str, upsert_sql: str,
                 kaynak: str, records: list, conn_params: dict = None,
                 cleanup_table: str = None, cleanup_create_sql: str = None) -> dict:
    """Shared body of the matched/review writers (same row shape, different table).

    cleanup_table: delete rows with the same (kaynak, record_id) from the
    other table in the same transaction. A record that went to review in one
    run and was resolved in the next used to stay in both tables.

    Returns {"basarili": True, "yazilan_kayit": N} or
    {"basarili": False, "sebep": "..."}; never raises.
    """
    if not records:
        return {"basarili": True, "yazilan_kayit": 0}

    try:
        import psycopg
    except ImportError:
        return {
            "basarili": False,
            "sebep": "psycopg kurulu degil. 'pip install -r requirements.txt' ile kurup tekrar deneyin.",
        }

    params = conn_params or _connection_params()
    try:
        with psycopg.connect(**params, connect_timeout=CONNECT_TIMEOUT_SECONDS) as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
                cur.execute(index_sql)
                rows = [_row_from_record(kaynak, r) for r in records]
                cur.executemany(upsert_sql, rows)
                if cleanup_table:
                    if cleanup_create_sql:
                        # the other table may not exist yet; a DELETE on a
                        # missing table would roll back the whole transaction
                        cur.execute(cleanup_create_sql)
                    record_ids = [r["record_id"] for r in records]
                    cur.execute(
                        f"DELETE FROM {cleanup_table} WHERE kaynak = %s AND record_id = ANY(%s);",
                        (kaynak, record_ids),
                    )
            conn.commit()
        return {"basarili": True, "yazilan_kayit": len(records)}
    except Exception as e:
        return {
            "basarili": False,
            "sebep": f"PostgreSQL'e yazilamadi ({params.get('host')}:{params.get('port')}/"
                     f"{params.get('dbname')}): {e}",
        }


def write_matched_records(kaynak: str, records: list, conn_params: dict = None) -> dict:
    """Upsert resolved records and drop stale copies from the review table."""
    return _write_rows(
        _CREATE_TABLE_SQL, _CREATE_INDEX_SQL, _UPSERT_SQL, kaynak, records, conn_params,
        cleanup_table=HUMAN_REVIEW_TABLE_NAME, cleanup_create_sql=_CREATE_HUMAN_REVIEW_TABLE_SQL,
    )


# ============================================================
# review queue (needs_human_review=true)
# ============================================================
HUMAN_REVIEW_TABLE_NAME = "human_review_kayitlari"

_CREATE_HUMAN_REVIEW_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {HUMAN_REVIEW_TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    kaynak TEXT NOT NULL,
    record_id TEXT NOT NULL,
    dominant_branch TEXT,
    overall_confidence NUMERIC(4,2),
    fields JSONB NOT NULL,
    guncellenme_zamani TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kaynak, record_id)
);
"""

_CREATE_HUMAN_REVIEW_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_{HUMAN_REVIEW_TABLE_NAME}_fields ON {HUMAN_REVIEW_TABLE_NAME} USING GIN (fields);
"""

_UPSERT_HUMAN_REVIEW_SQL = f"""
INSERT INTO {HUMAN_REVIEW_TABLE_NAME} (kaynak, record_id, dominant_branch, overall_confidence, fields, guncellenme_zamani)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (kaynak, record_id) DO UPDATE SET
    dominant_branch = EXCLUDED.dominant_branch,
    overall_confidence = EXCLUDED.overall_confidence,
    fields = EXCLUDED.fields,
    guncellenme_zamani = now();
"""


def write_human_review_records(kaynak: str, records: list, conn_params: dict = None) -> dict:
    """Upsert records waiting for review, unchanged, and drop stale copies
    from the matched table (a record that was resolved before and is
    ambiguous now)."""
    return _write_rows(
        _CREATE_HUMAN_REVIEW_TABLE_SQL, _CREATE_HUMAN_REVIEW_INDEX_SQL, _UPSERT_HUMAN_REVIEW_SQL,
        kaynak, records, conn_params,
        cleanup_table=TABLE_NAME, cleanup_create_sql=_CREATE_TABLE_SQL,
    )


# ============================================================
# canonical_units.node_id updates. The table is created by db/schema.sql and
# filled by scripts/seed_postgres.py; here we only UPDATE it.
# ============================================================


def update_canonical_unit_node_ids_bulk(entries: list, conn_params: dict = None) -> dict:
    """update_canonical_unit_node_id() for many entries, one connection and
    one transaction. entries: [{"kanonik_uuid", "node_id"}, ...]."""
    if not entries:
        return {"basarili": True, "guncellenen_kayit": 0}

    try:
        import psycopg
    except ImportError:
        return {
            "basarili": False,
            "sebep": "psycopg kurulu degil. 'pip install -r requirements.txt' ile kurup tekrar deneyin.",
        }

    params = conn_params or _connection_params()
    try:
        with psycopg.connect(**params, connect_timeout=CONNECT_TIMEOUT_SECONDS) as conn:
            with conn.cursor() as cur:
                # Check the unit exists before updating. rowcount can't tell
                # "already in the list" (0 rows, fine) from "no such unit".
                atlanan = []
                dogrulanan = []
                for entry in entries:
                    kanonik_uuid = entry["kanonik_uuid"]
                    cur.execute("SELECT 1 FROM canonical_units WHERE uuid = %s;", (kanonik_uuid,))
                    if cur.fetchone() is None:
                        atlanan.append(kanonik_uuid)
                        continue
                    dogrulanan.append(entry)
                for entry in dogrulanan:
                    node_id, kanonik_uuid = str(entry["node_id"]), entry["kanonik_uuid"]
                    cur.execute(
                        "UPDATE canonical_units SET node_id = array_remove(node_id, %s) "
                        "WHERE uuid <> %s AND %s = ANY(node_id);",
                        (node_id, kanonik_uuid, node_id),
                    )
                    cur.execute(
                        "UPDATE canonical_units SET node_id = array_append(node_id, %s) "
                        "WHERE uuid = %s AND NOT (%s = ANY(node_id));",
                        (node_id, kanonik_uuid, node_id),
                    )
            conn.commit()
        # nothing applied at all is a failure, not a partial success
        basarili = len(dogrulanan) > 0 or not atlanan
        sonuc = {"basarili": basarili, "guncellenen_kayit": len(dogrulanan)}
        if atlanan:
            sonuc["atlanan_kanonik_uuidler"] = atlanan
            sonuc["sebep"] = f"{len(atlanan)} kanonik_uuid canonical_units'te bulunamadi, atlandi."
        return sonuc
    except Exception as e:
        return {
            "basarili": False,
            "sebep": f"PostgreSQL'e yazilamadi ({params.get('host')}:{params.get('port')}/"
                     f"{params.get('dbname')}): {e}",
        }


def update_canonical_unit_node_id(kanonik_uuid: str, node_id: str, conn_params: dict = None) -> dict:
    """DB side of feedback.approve_node_id(): remove the node_id from every
    other unit and add it to this one, in one transaction. Returns
    basarili=False if the unit doesn't exist (an UPDATE matching no rows
    doesn't raise, so this has to be checked explicitly)."""
    try:
        import psycopg
    except ImportError:
        return {
            "basarili": False,
            "sebep": "psycopg kurulu degil. 'pip install -r requirements.txt' ile kurup tekrar deneyin.",
        }

    params = conn_params or _connection_params()
    try:
        with psycopg.connect(**params, connect_timeout=CONNECT_TIMEOUT_SECONDS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM canonical_units WHERE uuid = %s;", (kanonik_uuid,))
                if cur.fetchone() is None:
                    return {
                        "basarili": False,
                        "sebep": f"kanonik_uuid '{kanonik_uuid}' canonical_units'te bulunamadi.",
                    }
                cur.execute(
                    "UPDATE canonical_units SET node_id = array_remove(node_id, %s) "
                    "WHERE uuid <> %s AND %s = ANY(node_id);",
                    (node_id, kanonik_uuid, node_id),
                )
                cur.execute(
                    "UPDATE canonical_units SET node_id = array_append(node_id, %s) "
                    "WHERE uuid = %s AND NOT (%s = ANY(node_id));",
                    (node_id, kanonik_uuid, node_id),
                )
            conn.commit()
        return {"basarili": True}
    except Exception as e:
        return {
            "basarili": False,
            "sebep": f"PostgreSQL'e yazilamadi ({params.get('host')}:{params.get('port')}/"
                     f"{params.get('dbname')}): {e}",
        }


# ============================================================
# unit hierarchy paths (see engine/schema.py:full_hierarchy_chain)
# ============================================================
BIRIM_HIYERARSI_TABLE = "birim_hiyerarsi"

_CREATE_BIRIM_HIYERARSI_SQL = f"""
CREATE TABLE IF NOT EXISTS {BIRIM_HIYERARSI_TABLE} (
    uuid               UUID PRIMARY KEY,
    ad                 TEXT NOT NULL,
    kod                TEXT,
    zincir             JSONB NOT NULL,
    zincir_metni       TEXT NOT NULL,
    guncellenme_zamani TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_UPSERT_BIRIM_HIYERARSI_SQL = f"""
INSERT INTO {BIRIM_HIYERARSI_TABLE} (uuid, ad, kod, zincir, zincir_metni, guncellenme_zamani)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (uuid) DO UPDATE SET
    ad = EXCLUDED.ad, kod = EXCLUDED.kod, zincir = EXCLUDED.zincir,
    zincir_metni = EXCLUDED.zincir_metni, guncellenme_zamani = now();
"""


def write_birim_hiyerarsi(rows: list, conn_params: dict = None) -> dict:
    """Upsert each unit's full path to the root. Reference data derived from
    canonical_units, so upsert only, no cleanup."""
    if not rows:
        return {"basarili": True, "yazilan_kayit": 0}

    try:
        import psycopg
    except ImportError:
        return {
            "basarili": False,
            "sebep": "psycopg kurulu degil. 'pip install -r requirements.txt' ile kurup tekrar deneyin.",
        }

    params = conn_params or _connection_params()
    try:
        with psycopg.connect(**params, connect_timeout=CONNECT_TIMEOUT_SECONDS) as conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_BIRIM_HIYERARSI_SQL)
                data = [
                    (r["uuid"], r["ad"], r["kod"], json.dumps(r["zincir"], ensure_ascii=False), r["zincir_metni"])
                    for r in rows
                ]
                cur.executemany(_UPSERT_BIRIM_HIYERARSI_SQL, data)
            conn.commit()
        return {"basarili": True, "yazilan_kayit": len(rows)}
    except Exception as e:
        return {
            "basarili": False,
            "sebep": f"PostgreSQL'e yazilamadi ({params.get('host')}:{params.get('port')}/"
                     f"{params.get('dbname')}): {e}",
        }
