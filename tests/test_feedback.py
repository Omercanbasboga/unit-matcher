"""Human review loop: decisions, promotion, node_id approval, conflicting decisions."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.canonical_lookup import CanonicalPool
from engine.orchestrator import MatchEngine, FieldSpec
from engine.normalize import normalize
from engine import feedback


def check(name, condition, detail=""):
    assert condition, f"{name}: {detail}"


CANONICAL_FILES = ["canonical_units.json", "synonyms.json", "kadro_kurallari.json", "abbreviations.json"]


def _fresh_data_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="unit_matcher_test_"))
    for name in CANONICAL_FILES:
        shutil.copy(PROJECT_ROOT / "data" / name, tmp / name)
    return tmp


def _pick_two_real_units(pool: CanonicalPool):
    units = list(pool.units.values())
    return units[0], units[1]


def test_pool_loads_without_learned_files():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        check("Ogrenilen dosyalar yokken pool yukleniyor",
              pool.learned_corrections == {} and pool.learned_id_aliases == {})
        check("match_learned bilinmeyen metin icin None donuyor",
              pool.match_learned("OLMAYAN BIR BIRIM") is None)
        check("match_learned_id bilinmeyen id icin None donuyor",
              pool.match_learned_id("00000000") is None)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_text_based_learning_round_trip():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        unit, _ = _pick_two_real_units(pool)

        raw_text = "Tamamen Kirli ve Bilinmeyen Test Birim Adı XYZ123"
        norm_text = normalize(raw_text, data_dir)["normalized"]

        for _ in range(2):
            feedback.record_decision(
                data_dir, record_id="test-kayit-1", field_name="test_alan",
                karar_tipi=feedback.KARAR_TIPI_METIN, secilen_uuid=unit["uuid"],
                secilen_ad=unit["ad"], secilen_kod=unit["kod"],
                incoming_text=raw_text, normalized_text=norm_text,
                onceki_confidence=0.30, onceki_reason="test", reviewer="test-reviewer",
            )
        result = feedback.promote_to_learned_corrections(data_dir)
        check("Terfi edilen listede 1 kayit var", len(result["terfi_edilen"]) == 1, result)
        check("Celisen yok", len(result["celisen"]) == 0, result)

        pool2 = CanonicalPool(data_dir=data_dir)
        engine = MatchEngine(pool=pool2, llm_resolver=None)
        spec = FieldSpec(field_name="f", id_field_name="fid", incoming_text=raw_text, incoming_id=None)
        r = engine.resolve(spec).to_dict()
        check("Ogrenilen metin eslesmesi corrected_id dogru", str(r["corrected_id"]) == str(unit["kod"]), r)
        check("Ogrenilen metin eslesmesi emin_degil=False", r["emin_degil"] is False, r)
        check("Ogrenilen metin eslesmesi confidence 0.90-0.93 araliginda", 0.90 <= r["confidence"] <= 0.93, r)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_id_based_learning_round_trip():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        unit, _ = _pick_two_real_units(pool)

        fake_source_id = "fake-source-system-id-9999"
        feedback.record_decision(
            data_dir, record_id="test-kayit-2", field_name="job_fk_id_cozumu",
            karar_tipi=feedback.KARAR_TIPI_ID, secilen_uuid=unit["uuid"],
            secilen_ad=unit["ad"], secilen_kod=unit["kod"],
            incoming_id=fake_source_id, onceki_confidence=0.30, onceki_reason="test",
            reviewer="test-reviewer",
        )
        result = feedback.promote_to_learned_corrections(data_dir)
        check("ID-bazli terfi calisti", len(result["terfi_edilen"]) == 1, result)

        pool2 = CanonicalPool(data_dir=data_dir)
        engine = MatchEngine(pool=pool2, llm_resolver=None)
        r = engine.resolve_by_id("job_fk_id_cozumu", "job_fk_id", fake_source_id).to_dict()
        check("Ogrenilen ID eslesmesi corrected_id dogru", str(r["corrected_id"]) == str(unit["kod"]), r)
        check("Ogrenilen ID eslesmesi emin_degil=False", r["emin_degil"] is False, r)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


_FAKE_DB_CONN_PARAMS = {"host": "127.0.0.1", "port": "1", "dbname": "yok", "user": "yok", "password": ""}


def test_approve_node_id_yazar_ve_tam_guvenle_cozer():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        unit_a, _ = _pick_two_real_units(pool)

        fake_node_id = "test-node-id-8888"
        sonuc = feedback.approve_node_id(
            data_dir, node_id=fake_node_id, kanonik_uuid=unit_a["uuid"],
            secilen_ad=unit_a["ad"], secilen_kod=unit_a["kod"],
            reviewer="test-reviewer", conn_params=_FAKE_DB_CONN_PARAMS,
        )
        check("approve_node_id basarili (dosya tarafi)", sonuc["basarili"] is True, sonuc)
        check("DB tarafi zarifce basarisiz (fake port)", sonuc["db"]["basarili"] is False, sonuc)

        units = json.loads((data_dir / "canonical_units.json").read_text(encoding="utf-8"))
        check("node_id canonical_units.json'a yazildi",
              fake_node_id in units[unit_a["uuid"]]["node_id"], units[unit_a["uuid"]])

        log = json.loads((data_dir / "human_decisions_log.json").read_text(encoding="utf-8"))
        check("Audit izine yazildi", any(e.get("incoming_id") == fake_node_id for e in log), log)

        pool2 = CanonicalPool(data_dir=data_dir)
        check("by_alias_id'de dogru birime cozuluyor",
              pool2.by_alias_id.get(fake_node_id, {}).get("uuid") == unit_a["uuid"])

        engine = MatchEngine(pool=pool2, llm_resolver=None)
        r = engine.resolve_by_id("job_fk_id_cozumu", "job_fk_id", fake_node_id).to_dict()
        check("resolve_by_id emin_degil=False", r["emin_degil"] is False, r)
        check("resolve_by_id corrected_id dogru", str(r["corrected_id"]) == str(unit_a["kod"]), r)
        check("resolve_by_id confidence 0.97 (kaynak-beyani ile ayni seviye, "
              "learned_id_aliases'in dusuk-guvenli 0.90-0.93'u DEGIL)",
              r["confidence"] == 0.97, r)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_approve_node_id_baska_birimden_tasir():
    """Approving an ID for one unit removes it from the unit that had it before."""
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        unit_a, unit_b = _pick_two_real_units(pool)

        node_id = "test-node-id-tasima-7777"
        feedback.approve_node_id(
            data_dir, node_id=node_id, kanonik_uuid=unit_a["uuid"],
            reviewer="test-reviewer", conn_params=_FAKE_DB_CONN_PARAMS,
        )
        feedback.approve_node_id(
            data_dir, node_id=node_id, kanonik_uuid=unit_b["uuid"],
            reviewer="test-reviewer", conn_params=_FAKE_DB_CONN_PARAMS,
        )

        units = json.loads((data_dir / "canonical_units.json").read_text(encoding="utf-8"))
        check("Eski birimden silindi", node_id not in units[unit_a["uuid"]]["node_id"], units[unit_a["uuid"]])
        check("Yeni birime eklendi", node_id in units[unit_b["uuid"]]["node_id"], units[unit_b["uuid"]])

        pool2 = CanonicalPool(data_dir=data_dir)
        check("by_alias_id artik YENI birimi gosteriyor",
              pool2.by_alias_id.get(node_id, {}).get("uuid") == unit_b["uuid"])
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_conflicting_decisions_not_promoted():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        unit_a, unit_b = _pick_two_real_units(pool)

        raw_text = "Baska Bir Celisen Test Birim Adı ABC456"
        norm_text = normalize(raw_text, data_dir)["normalized"]

        feedback.record_decision(
            data_dir, record_id="k1", field_name="f", karar_tipi=feedback.KARAR_TIPI_METIN,
            secilen_uuid=unit_a["uuid"], secilen_ad=unit_a["ad"], secilen_kod=unit_a["kod"],
            normalized_text=norm_text, reviewer="reviewer-a",
        )
        feedback.record_decision(
            data_dir, record_id="k2", field_name="f", karar_tipi=feedback.KARAR_TIPI_METIN,
            secilen_uuid=unit_b["uuid"], secilen_ad=unit_b["ad"], secilen_kod=unit_b["kod"],
            normalized_text=norm_text, reviewer="reviewer-b",
        )
        result = feedback.promote_to_learned_corrections(data_dir)
        check("Celisen kayit tespit edildi", len(result["celisen"]) == 1, result)
        check("Celisen kayit terfi etmedi", len(result["terfi_edilen"]) == 0, result)

        learned = json.loads((data_dir / "learned_corrections.json").read_text(encoding="utf-8"))
        check("Celisen metin learned_corrections.json'a yazilmadi", norm_text not in learned, learned)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_conflicting_decision_retracts_stale_promotion():
    """Three reviewers said A, a fourth says B: the earlier promotion of A is taken back."""
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        unit_a, unit_b = _pick_two_real_units(pool)

        raw_text = "Once Dogru Sanilan Sonra Celisen Test Birim Adı DEF789"
        norm_text = normalize(raw_text, data_dir)["normalized"]

        feedback.record_decision(
            data_dir, record_id="k1", field_name="f", karar_tipi=feedback.KARAR_TIPI_METIN,
            secilen_uuid=unit_a["uuid"], secilen_ad=unit_a["ad"], secilen_kod=unit_a["kod"],
            normalized_text=norm_text, reviewer="reviewer-a",
        )
        result1 = feedback.promote_to_learned_corrections(data_dir)
        check("1. tur: A'ya terfi etti", len(result1["terfi_edilen"]) == 1, result1)
        learned1 = json.loads((data_dir / "learned_corrections.json").read_text(encoding="utf-8"))
        check("1. tur: learned_corrections.json'da A var", learned1.get(norm_text, {}).get("kanonik_uuid") == unit_a["uuid"], learned1)

        feedback.record_decision(
            data_dir, record_id="k2", field_name="f", karar_tipi=feedback.KARAR_TIPI_METIN,
            secilen_uuid=unit_b["uuid"], secilen_ad=unit_b["ad"], secilen_kod=unit_b["kod"],
            normalized_text=norm_text, reviewer="reviewer-b",
        )
        result2 = feedback.promote_to_learned_corrections(data_dir)
        check("2. tur: celiski tespit edildi", len(result2["celisen"]) == 1, result2)
        check("2. tur: celiski 'geri_alindi=True' ile isaretlendi",
              result2["celisen"][0].get("geri_alindi") is True, result2)
        check("2. tur: bu turda YENI bir terfi olmadi", len(result2["terfi_edilen"]) == 0, result2)

        learned2 = json.loads((data_dir / "learned_corrections.json").read_text(encoding="utf-8"))
        check("2. tur: eski (artik supheli) A girdisi learned_corrections.json'dan SILINDI",
              norm_text not in learned2, learned2)

        pool2 = CanonicalPool(data_dir=data_dir)
        engine = MatchEngine(pool=pool2, llm_resolver=None)
        spec = FieldSpec(field_name="f", id_field_name="fid", incoming_text=raw_text, incoming_id=None)
        r = engine.resolve(spec).to_dict()
        check("Motor artik bu metni ogrenilmis-eslesme olarak KOR GUVENLE kabul etmiyor",
              r["adim"] != "kod_hitl_ogrenilmis_metin", r)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_missing_canonical_report():
    data_dir = _fresh_data_dir()
    try:
        feedback.report_missing_canonical(
            data_dir, record_id="k3", field_name="f", incoming_text="Hic Yok Boyle Bir Test Birimi",
            normalized_text="HİÇ YOK BÖYLE BİR TEST BİRİMİ", onceki_reason="test", reviewer="reviewer-c",
        )
        reports = json.loads((data_dir / "kanonik_eksik_bildirimleri.json").read_text(encoding="utf-8"))
        check("Kanonik eksik bildirimi kaydedildi", len(reports) == 1, reports)

        result = feedback.promote_to_learned_corrections(data_dir)
        check("Eksik-veri bildirimi promote'u etkilemiyor", result["terfi_edilen"] == [], result)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
