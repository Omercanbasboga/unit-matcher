"""End-to-end decision tree scenarios against the sample data."""
import atexit
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.canonical_lookup import CanonicalPool
from engine.orchestrator import MatchEngine, FieldSpec
from engine.schema import build_record_output, full_hierarchy_chain
from engine.normalize import normalize
from engine.hierarchy import HierarchyEngine, violates_veto

_TEMP_DATA_DIR = Path(tempfile.mkdtemp(prefix="unit_matcher_test_scenarios_"))
for _name in ["canonical_units.json", "synonyms.json", "kadro_kurallari.json", "abbreviations.json"]:
    shutil.copy(PROJECT_ROOT / "data" / _name, _TEMP_DATA_DIR / _name)
atexit.register(shutil.rmtree, _TEMP_DATA_DIR, ignore_errors=True)

POOL = CanonicalPool(data_dir=_TEMP_DATA_DIR)
ENGINE_NO_LLM = MatchEngine(pool=POOL, llm_resolver=None)


def check(name, condition, detail=""):
    assert condition, f"{name}: {detail}"


def test_kadro_ihlali_insana_duser():
    """A professor can't be placed in the IT department: leave the field alone and flag it."""
    spec = FieldSpec(field_name="personelBirim", id_field_name="personelBirimid",
                      incoming_text="Bilgi İşlem Daire Başkanlığı", incoming_id="166",
                      unvan="Profesör Dr.")
    r = ENGINE_NO_LLM.resolve(spec).to_dict()
    check("Senaryo D: changed=False", r["changed"] is False, r)
    check("Senaryo D: emin_degil=True", r["emin_degil"] is True, r)
    check("Senaryo D: confidence < 0.50", r["confidence"] < 0.50, r)


def test_kisaltma_acilip_tam_eslesir():
    """"Yabancı Diller Yo" -> abbreviation expanded -> exact match."""
    spec = FieldSpec(field_name="gelen_birim_adi", id_field_name="abd_fk_id",
                      incoming_text="Yabancı Diller Yo", incoming_id="3c9e1a52-7f04-4d6b-a2e8-91b0c4f6d7a3",
                      jobrecord_tipi="Akademik")
    r = ENGINE_NO_LLM.resolve(spec).to_dict()
    check("Senaryo H: corrected_text doğru", r["corrected_text"] == "YABANCI DİLLER YÜKSEKOKULU", r)
    check("Senaryo H: corrected_id doğru", str(r["corrected_id"]) == "156", r)
    check("Senaryo H: changed=True", r["changed"] is True, r)
    check("Senaryo H: emin_degil=False", r["emin_degil"] is False, r)


def test_llm_secimi_uygulanir():
    """Two equally close candidates: the LLM's pick is used; without an LLM it goes to review."""

    def fake_llm(norm_text, candidates, spec):
        for c in candidates:
            if c["ad"] == "SOSYAL BİLİMLER":
                return {"secilen_uuid": c["uuid"], "confidence": 0.9, "reason": "test"}
        return None

    engine_with_llm = MatchEngine(pool=POOL, llm_resolver=fake_llm)
    spec = FieldSpec(field_name="gelen_birim_adi", id_field_name="job_fk_id",
                      incoming_text="Sosyal Bilimler Ens. Varyasyon",
                      incoming_id="e2b7d410-58ca-4f39-9d1e-6a4c0b8f2e95",
                      unvan="Profesör Dr.", jobrecord_tipi="Akademik")
    r = engine_with_llm.resolve(spec).to_dict()
    check("Senaryo I: LLM secimi uygulandi", r["corrected_text"] == "SOSYAL BİLİMLER", r)
    check("Senaryo I: corrected_id doğru", str(r["corrected_id"]) == "5500", r)
    check("Senaryo I: changed=True", r["changed"] is True, r)

    r_no_llm = ENGINE_NO_LLM.resolve(spec).to_dict()
    check("Senaryo I (LLM'siz): reddedildi ama crash yok", r_no_llm["emin_degil"] is True, r_no_llm)


def test_adim1_senonim_kendi_kendini_zehirleme_regresyonu():
    """An ID learned from a synonym match must still resolve on the next record with the same short text."""
    test_id = "8887771"
    spec1 = FieldSpec(field_name="birim_ad", id_field_name="birim_id",
                       incoming_text="BAP Koordinatörlüğü", incoming_id=test_id)
    r1 = ENGINE_NO_LLM.resolve(spec1).to_dict()
    check("1. gecis: senonimle cozuldu", r1["emin_degil"] is False, r1)
    check("1. gecis: dogru resmi isme cozuldu",
          r1["corrected_text"] == "BİLİMSEL ARAŞTIRMA PROJELERİ KOORDİNATÖRLÜĞÜ", r1)
    check("1. gecis: ID otomatik ogrenildi (by_alias_id'ye eklendi)",
          POOL.by_alias_id.get(test_id) is not None, POOL.by_alias_id.get(test_id))

    spec2 = FieldSpec(field_name="birim_ad", id_field_name="birim_id",
                       incoming_text="BAP Koordinatörlüğü", incoming_id=test_id)
    r2 = ENGINE_NO_LLM.resolve(spec2).to_dict()
    check("2. gecis: artik Adim 1'de senonim sayesinde DOGRU cozuldu (insan onayina DUSMEDI)",
          r2["emin_degil"] is False, r2)
    check("2. gecis: Adim 1'den (ID-oncelikli) cozuldu", r2["adim"] == "kod_id_crosswalk_oncelikli", r2)
    check("2. gecis: dogru resmi isme cozuldu",
          r2["corrected_text"] == "BİLİMSEL ARAŞTIRMA PROJELERİ KOORDİNATÖRLÜĞÜ", r2)


def test_veto_daire_baskanligi_vs_mudurluk():
    """A MÜDÜRLÜĞÜ is never rounded to the DAİRE BAŞKANLIĞI with almost the same name."""
    ihlal_var = violates_veto("BİLGİ İŞLEM DAİRE BAŞKANLIĞI", "BİLGİ İŞLEM MÜDÜRLÜĞÜ")
    check("Veto: DAİRE BAŞKANLIĞI <-> MÜDÜRLÜĞÜ ihlali tespit edildi", ihlal_var)

    spec = FieldSpec(field_name="x", id_field_name="y",
                      incoming_text="Sağlık Kültür ve Spor Müdürlüğü", incoming_id="800")
    r = ENGINE_NO_LLM.resolve(spec).to_dict()
    check("Veto: motor DAİRE BAŞKANLIĞI'na yuvarlamadi",
          r["corrected_text"] != "SAĞLIK KÜLTÜR VE SPOR DAİRE BAŞKANLIĞI", r)
    check("Veto: bulunan aday dogru seviyede (MÜDÜRLÜĞÜ)",
          r["corrected_text"].endswith("MÜDÜRLÜĞÜ"), r)


def test_synonym_id_mismatch_detection():
    """The synonym table's ID differs from the unit's org code; both are reported."""
    norm = normalize("BAP Koordinatörlüğü")["normalized"]
    syn = POOL.match_synonym(norm)
    check("Senonim ID uyumsuzlugu tespit edildi", syn["id_uyusmazligi"] is True, syn)
    check("Senonim org_unit yine de dogru bulundu (kod=180)",
          syn["org_unit"] is not None and syn["org_unit"]["kod"] == "180", syn)


def test_hierarchy_scoped_parent_resolution():
    """Parent text with an extra BÖLÜMÜ still finds the unit stored without it."""
    hz = HierarchyEngine(POOL)
    parent = hz.resolve_parent_unit("Bilgisayar Mühendisliği Bölümü")
    check("Hiyerarsi: esnek ek-cikarma ile parent bulundu",
          parent is not None and parent["kod"] == "4901", parent)


def test_hierarchy_parent_pool_includes_parent_itself():
    """Parent text equal to the unit's own name: the unit itself must be a candidate."""
    spec = FieldSpec(field_name="birim_ad", id_field_name="birim_id",
                      incoming_text="TAHAKKUK ŞUBE MÜDÜRLÜĞÜ", incoming_id="999999",
                      parent_text="TAHAKKUK ŞUBE MÜDÜRLÜĞÜ")
    r = ENGINE_NO_LLM.resolve(spec).to_dict()
    check("Parent=kendisi: emin_degil=False (artik cozuluyor)", r["emin_degil"] is False, r)
    check("Parent=kendisi: corrected_id=967 (TAHAKKUK ŞUBE MÜDÜRLÜĞÜ)", r["corrected_id"] == "967", r)
    check("Parent=kendisi: confidence yuksek (birebir dize eslesmesi)", r["confidence"] >= 0.95, r)


def test_changed_bayragi_turkce_i_hassasiyeti():
    """Same text in different case (with Turkish i) is not a change."""
    unit = {"kod": "99999", "ad": "İDARİ TEST BİRİMİ", "tur": None, "uuid": "test-uuid-tr-i"}
    spec = FieldSpec(field_name="test_alan", id_field_name="test_id",
                      incoming_text="idari test birimi", incoming_id="99999")
    sonuc = ENGINE_NO_LLM._apply_kadro_and_finish(  # noqa: SLF001
        spec, unit, 0.95, "test", source_layer="kod", adim="test_adim",
    )
    check("Turkce 'i' iceren ama ICERIK olarak AYNI metin changed=False olmali",
          sonuc.changed is False, sonuc.to_dict())


def test_corrupt_input_rejected_without_crash():
    spec = FieldSpec(field_name="x", id_field_name="y",
                      incoming_text="qwertyasdfgh12345", incoming_id="0")
    r = ENGINE_NO_LLM.resolve(spec).to_dict()
    check("Cop veri: changed=False", r["changed"] is False, r)
    check("Cop veri: confidence < 0.50", r["confidence"] < 0.50, r)


def test_record_level_schema_aggregation():
    """Record confidence is the weakest field; one bad field sends the record to review."""
    spec_ok = FieldSpec(field_name="a", id_field_name="a_id",
                         incoming_text="Eğitim Fakültesi", incoming_id="3200")
    spec_bad = FieldSpec(field_name="b", id_field_name="b_id",
                          incoming_text="qwertyasdfgh12345", incoming_id="0")
    r_ok = ENGINE_NO_LLM.resolve(spec_ok)
    r_bad = ENGINE_NO_LLM.resolve(spec_bad)
    record = build_record_output("test-1", [r_ok, r_bad], pool=POOL)
    check("Record: needs_human_review=True (bir alan basarisiz oldugu icin)",
          record["needs_human_review"] is True, record)
    check("Record: overall_confidence = min(alanlar)",
          record["overall_confidence"] == min(r_ok.confidence, r_bad.confidence), record)
    check("Record: dominant_branch dolu (en az bir alan cozuldugu icin)",
          record["dominant_branch"] is not None, record)


def test_full_hierarchy_chain_dogrudan_rektorluge_bagli_birim():
    unit = POOL.by_kod.get("156")
    chain = full_hierarchy_chain(POOL, unit)
    check("Zincir 2 seviyeli (REKTÖRLÜK + kendisi)", len(chain["zincir"]) == 2, chain)
    check("Zincir REKTÖRLÜK ile basliyor", chain["zincir"][0]["ad"] == "REKTÖRLÜK", chain)
    check("Zincir kendisiyle bitiyor", chain["zincir"][-1]["uuid"] == unit["uuid"], chain)
    check("zincir_metni dogru formatlanmis",
          chain["zincir_metni"] == "REKTÖRLÜK / YABANCI DİLLER YÜKSEKOKULU", chain)


def test_full_hierarchy_chain_cok_seviyeli_birim():
    unit = POOL.by_kod.get("839")
    chain = full_hierarchy_chain(POOL, unit)
    ad_sirasi = [u["ad"] for u in chain["zincir"]]
    check("Zincir REKTÖRLÜK ile basliyor", ad_sirasi[0] == "REKTÖRLÜK", ad_sirasi)
    check("Zincir kendisiyle bitiyor", ad_sirasi[-1] == unit["ad"], ad_sirasi)
    check("Zincir en az 3 seviyeli (ara kademeler korunmus)", len(ad_sirasi) >= 3, ad_sirasi)
    check("zincir_metni ' / ' ile birlestirilmis ve ayni sirada",
          chain["zincir_metni"] == " / ".join(ad_sirasi), chain)
