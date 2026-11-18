"""Postgres writer: pure helpers, and failing quietly when there's no database."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.db_sink import (
    _row_from_record,
    write_matched_records,
    write_birim_hiyerarsi,
    update_canonical_unit_node_id,
)


def check(name, condition, detail=""):
    assert condition, f"{name}: {detail}"


def test_row_from_record_shape():
    record = {
        "record_id": "abc-123",
        "fields": [{"field_name": "f", "corrected_id": "156"}],
        "dominant_branch": "REKTÖRLÜK / YABANCI DİLLER YÜKSEKOKULU",
        "overall_confidence": 0.95,
        "needs_human_review": False,
    }
    row = _row_from_record("json_sonuc", record)
    check("Demet uzunlugu 5", len(row) == 5, row)
    check("kaynak dogru", row[0] == "json_sonuc", row)
    check("record_id dogru", row[1] == "abc-123", row)
    check("dominant_branch dogru", row[2] == record["dominant_branch"], row)
    check("overall_confidence dogru", row[3] == 0.95, row)
    check("fields JSON string olarak kodlanmis", json.loads(row[4]) == record["fields"], row)


def test_write_matched_records_empty_list_no_connection_attempt():
    result = write_matched_records("json_sonuc", [])
    check("Bos liste basarili donuyor", result == {"basarili": True, "yazilan_kayit": 0}, result)


def test_write_matched_records_graceful_failure():
    fake_record = {
        "record_id": "test-1", "fields": [], "dominant_branch": None,
        "overall_confidence": 0.9, "needs_human_review": False,
    }
    try:
        result = write_matched_records(
            "json_sonuc", [fake_record],
            conn_params={"host": "127.0.0.1", "port": "1", "dbname": "yok", "user": "yok", "password": ""},
        )
        exploded = False
    except Exception:
        exploded = True
        result = None

    check("write_matched_records exception firlatmadi", exploded is False, result)
    check("basarisiz sonuc dondu", result is not None and result["basarili"] is False, result)
    check("sebep alani dolu", bool(result and result.get("sebep")), result)


def test_write_birim_hiyerarsi_empty_list_no_connection_attempt():
    result = write_birim_hiyerarsi([])
    check("Bos liste basarili donuyor", result == {"basarili": True, "yazilan_kayit": 0}, result)


def test_write_birim_hiyerarsi_graceful_failure():
    fake_row = {
        "uuid": "00000000-0000-0000-0000-000000000000", "ad": "TEST", "kod": "1",
        "zincir": [{"uuid": "x", "ad": "REKTÖRLÜK", "kod": "100"}], "zincir_metni": "REKTÖRLÜK / TEST",
    }
    try:
        result = write_birim_hiyerarsi(
            [fake_row],
            conn_params={"host": "127.0.0.1", "port": "1", "dbname": "yok", "user": "yok", "password": ""},
        )
        exploded = False
    except Exception:
        exploded = True
        result = None

    check("write_birim_hiyerarsi exception firlatmadi", exploded is False, result)
    check("basarisiz sonuc dondu", result is not None and result["basarili"] is False, result)
    check("sebep alani dolu", bool(result and result.get("sebep")), result)


def test_update_canonical_unit_node_id_graceful_failure():
    try:
        result = update_canonical_unit_node_id(
            "00000000-0000-0000-0000-000000000000", "test-node-id",
            conn_params={"host": "127.0.0.1", "port": "1", "dbname": "yok", "user": "yok", "password": ""},
        )
        exploded = False
    except Exception:
        exploded = True
        result = None

    check("update_canonical_unit_node_id exception firlatmadi", exploded is False, result)
    check("basarisiz sonuc dondu", result is not None and result["basarili"] is False, result)
    check("sebep alani dolu", bool(result and result.get("sebep")), result)
