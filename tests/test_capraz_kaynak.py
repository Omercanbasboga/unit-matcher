"""ID handling: known IDs vs. text, learned IDs, cross-source conflict notes, raw record in the prompt."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.canonical_lookup import CanonicalPool
from engine.orchestrator import MatchEngine, FieldSpec
from engine.llm_fallback import build_prompt


def check(name, condition, detail=""):
    assert condition, f"{name}: {detail}"


CANONICAL_FILES = ["canonical_units.json", "synonyms.json", "kadro_kurallari.json", "abbreviations.json"]


def _fresh_data_dir(capraz_kaynak_uyarilari: dict = None) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="unit_matcher_test_"))
    for name in CANONICAL_FILES:
        shutil.copy(PROJECT_ROOT / "data" / name, tmp / name)
    if capraz_kaynak_uyarilari is not None:
        with open(tmp / "capraz_kaynak_id_uyarilari.json", "w", encoding="utf-8") as f:
            json.dump(capraz_kaynak_uyarilari, f, ensure_ascii=False)
    return tmp


def _find_clean_node_id_unit(units: dict):
    kod_set = {u["kod"] for u in units.values()}
    for unit in units.values():
        for alias in unit.get("node_id", []):
            if alias not in kod_set:
                return unit, alias
    return None, None


def test_pool_loads_without_capraz_kaynak_dosyasi():
    """The conflict snapshot file is optional."""
    data_dir = _fresh_data_dir(capraz_kaynak_uyarilari=None)
    try:
        pool = CanonicalPool(data_dir=data_dir)
        check("Dosya yokken pool yukleniyor", pool is not None)
        check("id_celiskileri bos", pool._id_celiskileri == {})
        check("birim_coklu_id bos", pool._birim_coklu_id == {})
        check("flagged_incoming_id_warning None donuyor", pool.flagged_incoming_id_warning("999") is None)
    finally:
        shutil.rmtree(data_dir)


def test_id_celiskisi_tespit_ediliyor():
    uyarilar = {
        "id_celiskileri": {
            "1240": [
                {"kaynak": "idari_gorev", "corrected_id": "750", "corrected_text": "PERSONEL DAİRE BAŞKANLIĞI", "gorulme_sayisi": 3},
                {"kaynak": "fiili_gorev", "corrected_id": "105", "corrected_text": "GENEL SEKRETERLİK", "gorulme_sayisi": 1},
            ]
        },
        "birim_coklu_id": {},
    }
    data_dir = _fresh_data_dir(uyarilar)
    try:
        pool = CanonicalPool(data_dir=data_dir)
        uyari = pool.flagged_incoming_id_warning("1240")
        check("Celiskili ID icin uyari donuyor", uyari is not None)
        check("Uyari her iki kaynagi da iceriyor", "idari_gorev" in uyari and "fiili_gorev" in uyari, uyari)
        check("Uyari her iki birimi de iceriyor", "PERSONEL DAİRE BAŞKANLIĞI" in uyari and "GENEL SEKRETERLİK" in uyari, uyari)
        check("Celiskisiz ID icin None donuyor", pool.flagged_incoming_id_warning("777777") is None)
        check("int olarak verilse bile eslesiyor", pool.flagged_incoming_id_warning(1240) is not None)
    finally:
        shutil.rmtree(data_dir)


def test_birim_coklu_id_tespit_ediliyor():
    uyarilar = {
        "id_celiskileri": {},
        "birim_coklu_id": {
            "3200": {
                "corrected_text": "EĞİTİM FAKÜLTESİ",
                "gelen_idler": [
                    {"kaynak": "idari_gorev", "incoming_id": "1056", "gorulme_sayisi": 5},
                    {"kaynak": "canli_personel", "incoming_id": "134", "gorulme_sayisi": 2},
                ],
            }
        },
    }
    data_dir = _fresh_data_dir(uyarilar)
    try:
        pool = CanonicalPool(data_dir=data_dir)
        uyari_1 = pool.flagged_incoming_id_warning("1056")
        uyari_2 = pool.flagged_incoming_id_warning("134")
        check("Coklu-ID birimin ilk ID'si icin uyari donuyor", uyari_1 is not None)
        check("Coklu-ID birimin ikinci ID'si icin de uyari donuyor", uyari_2 is not None)
        check("Uyari birim adini iceriyor", "EĞİTİM FAKÜLTESİ" in uyari_1, uyari_1)
        check("Ilgisiz ID icin None donuyor", pool.flagged_incoming_id_warning("999999") is None)
    finally:
        shutil.rmtree(data_dir)


def test_resolve_unknown_id_eski_statik_raporu_artik_kullanmiyor():
    """The snapshot is informational only; resolve() doesn't block on it and still learns the ID."""
    uyarilar = {
        "id_celiskileri": {"9999": [{"kaynak": "idari_gorev", "corrected_id": "180", "corrected_text": "BİLİMSEL ARAŞTIRMA PROJELERİ KOORDİNATÖRLÜĞÜ", "gorulme_sayisi": 1}]},
        "birim_coklu_id": {},
    }
    data_dir = _fresh_data_dir(uyarilar)
    try:
        pool = CanonicalPool(data_dir=data_dir)
        llm_cagrildi = {"flag": False}

        def sahte_llm_resolver(norm_text, candidates, spec):
            llm_cagrildi["flag"] = True
            return None

        engine = MatchEngine(pool=pool, llm_resolver=sahte_llm_resolver)
        spec = FieldSpec(
            field_name="birim_ad", id_field_name="birim_id",
            incoming_text="BAP KOORDİNATÖRLÜĞÜ", incoming_id="9999",
        )
        r = engine.resolve(spec).to_dict()
        check("Eski rapor artik etkisiz: emin_degil=False", r["emin_degil"] is False, r)
        check("Eski rapor artik etkisiz: senonim normal calisti",
              r["corrected_text"] == "BİLİMSEL ARAŞTIRMA PROJELERİ KOORDİNATÖRLÜĞÜ", r)
        check("llm_resolver HIC cagrilmadi (senonim yetti)", llm_cagrildi["flag"] is False)
        check("Yeni ID otomatik ogrenme kuyruguna eklendi",
              any(e["node_id"] == "9999" for e in engine.auto_learned_node_ids),
              engine.auto_learned_node_ids)
    finally:
        shutil.rmtree(data_dir)


def test_resolve_flagged_olmayan_id_normal_calisiyor():
    uyarilar = {"id_celiskileri": {"9999": [{}]}, "birim_coklu_id": {}}
    data_dir = _fresh_data_dir(uyarilar)
    try:
        pool = CanonicalPool(data_dir=data_dir)
        engine = MatchEngine(pool=pool, llm_resolver=None)
        spec = FieldSpec(
            field_name="birim_ad", id_field_name="birim_id",
            incoming_text="BAP KOORDİNATÖRLÜĞÜ", incoming_id="1234",  # not flagged
        )
        r = engine.resolve(spec).to_dict()
        check("Flagged olmayan ID: emin_degil=False", r["emin_degil"] is False, r)
        check("Flagged olmayan ID: senonim normal calisti",
              r["corrected_text"] == "BİLİMSEL ARAŞTIRMA PROJELERİ KOORDİNATÖRLÜĞÜ", r)
    finally:
        shutil.rmtree(data_dir)


def test_resolve_by_id_bilinmeyen_id_kanonik_havuzda_yok_diye_reddediyor():
    uyarilar = {
        "id_celiskileri": {"5555": [{"kaynak": "a", "corrected_id": "1", "corrected_text": "X", "gorulme_sayisi": 1}]},
        "birim_coklu_id": {},
    }
    data_dir = _fresh_data_dir(uyarilar)
    try:
        pool = CanonicalPool(data_dir=data_dir)
        engine = MatchEngine(pool=pool, llm_resolver=None)
        r = engine.resolve_by_id("birim_ad", "birim_id", "5555").to_dict()
        check("resolve_by_id bilinmeyen ID: emin_degil=True", r["emin_degil"] is True, r)
        check("resolve_by_id bilinmeyen ID: reason 'kanonik havuzda bulunamadi' diyor",
              "kanonik havuzda bulunamadi" in r["reason"], r["reason"])
        check("resolve_by_id bilinmeyen ID: Capraz Kaynak Kontrolu ARTIK bahsedilmiyor",
              "Capraz Kaynak Kontrolu" not in r["reason"], r["reason"])
    finally:
        shutil.rmtree(data_dir)


def test_resolve_bilinen_id_isim_tutarsizsa_reddediyor():
    """Known ID, unrelated text: the ID is ignored, the text fails too, and the reason says both."""
    data_dir = _fresh_data_dir(capraz_kaynak_uyarilari=None)
    try:
        with open(data_dir / "canonical_units.json", encoding="utf-8") as f:
            units = json.load(f)
        target_unit, bilinen_node_id = _find_clean_node_id_unit(units)
        check("Test verisinde node_id'li en az bir birim var", target_unit is not None)

        pool = CanonicalPool(data_dir=data_dir)
        engine = MatchEngine(pool=pool, llm_resolver=None)
        r = engine.resolve(FieldSpec(
            field_name="birim_ad", id_field_name="birim_id",
            incoming_text="TAMAMEN ALAKASIZ RASTGELE BIR METIN QWERTY",
            incoming_id=bilinen_node_id,
        )).to_dict()
        check("ID/isim celiskisi: emin_degil=True", r["emin_degil"] is True, r)
        check("ID/isim celiskisi: changed=False", r["changed"] is False, r)
        check("ID/isim celiskisi: reason ID celiskisinin yok sayildigini belirtiyor",
              "ID yok sayilip metin uzerinden devam edildi" in r["reason"], r["reason"])
        check("ID/isim celiskisi: reason bilinen birimin adini iceriyor",
              target_unit["ad"] in r["reason"], r["reason"])
    finally:
        shutil.rmtree(data_dir)


def test_resolve_bilinen_id_isim_tutarliysa_kabul_ediyor():
    """Known ID and matching text: accepted at 0.97 without calling the LLM."""
    data_dir = _fresh_data_dir(capraz_kaynak_uyarilari=None)
    try:
        with open(data_dir / "canonical_units.json", encoding="utf-8") as f:
            units = json.load(f)
        target_unit, bilinen_node_id = _find_clean_node_id_unit(units)
        check("Test verisinde node_id'li en az bir birim var", target_unit is not None)

        pool = CanonicalPool(data_dir=data_dir)
        llm_cagrildi = {"flag": False}

        def sahte_llm_resolver(norm_text, candidates, spec):
            llm_cagrildi["flag"] = True
            return None

        engine = MatchEngine(pool=pool, llm_resolver=sahte_llm_resolver)
        r = engine.resolve(FieldSpec(
            field_name="birim_ad", id_field_name="birim_id",
            incoming_text=target_unit["ad"], incoming_id=bilinen_node_id,
        )).to_dict()
        check("ID+isim tutarli: emin_degil=False", r["emin_degil"] is False, r)
        check("ID+isim tutarli: confidence 0.97", abs(r["confidence"] - 0.97) < 1e-6, r)
        check("ID+isim tutarli: dogru birime cozuldu",
              str(r["corrected_id"]) == str(target_unit["kod"]), r)
        check("ID+isim tutarli: llm_resolver HIC cagrilmadi", llm_cagrildi["flag"] is False)
    finally:
        shutil.rmtree(data_dir)


def test_bilinen_node_id_gate_i_atlar():
    """A known alias resolves by ID even when the snapshot flags it."""
    data_dir = _fresh_data_dir(capraz_kaynak_uyarilari=None)
    try:
        with open(data_dir / "canonical_units.json", encoding="utf-8") as f:
            units = json.load(f)
        target_unit, bilinen_node_id = _find_clean_node_id_unit(units)
        check("Test verisinde node_id'li en az bir birim var", target_unit is not None)

        uyarilar = {
            "id_celiskileri": {
                bilinen_node_id: [
                    {"kaynak": "idari_gorev", "corrected_id": "999", "corrected_text": "BASKA BIR BIRIM", "gorulme_sayisi": 1},
                ]
            },
            "birim_coklu_id": {
                target_unit["kod"]: {
                    "corrected_text": target_unit["ad"],
                    "gelen_idler": [{"kaynak": "fiili_gorev", "incoming_id": bilinen_node_id, "gorulme_sayisi": 3}],
                }
            },
        }
        with open(data_dir / "capraz_kaynak_id_uyarilari.json", "w", encoding="utf-8") as f:
            json.dump(uyarilar, f, ensure_ascii=False)

        pool = CanonicalPool(data_dir=data_dir)
        check("On kosul: bu ID capraz-kaynak dosyasinda flagged",
              pool.flagged_incoming_id_warning(bilinen_node_id) is not None)
        check("is_known_incoming_id True donuyor", pool.is_known_incoming_id(bilinen_node_id) is True)

        engine = MatchEngine(pool=pool, llm_resolver=None)
        r_id = engine.resolve_by_id("birim_ad", "birim_id", bilinen_node_id).to_dict()
        check("resolve_by_id: bilinen node_id reddedilmedi (emin_degil=False)",
              r_id["emin_degil"] is False, r_id)
        check("resolve_by_id: dogru birime cozuldu",
              str(r_id["corrected_id"]) == str(target_unit["kod"]), r_id)
    finally:
        shutil.rmtree(data_dir)


def test_alias_baska_birimin_resmi_koduyla_cakisirsa_resmi_kod_kazanir():
    """"652" is an old alias of one unit and the official code of another. The official code wins."""
    data_dir = _fresh_data_dir(capraz_kaynak_uyarilari=None)
    try:
        with open(data_dir / "canonical_units.json", encoding="utf-8") as f:
            units = json.load(f)
        kod_owner = next(u for u in units.values() if u["kod"] == "652")
        alias_claimant = next(
            u for u in units.values()
            if "652" in u.get("node_id", []) and u["kod"] != "652"
        )
        check("Test on-kosulu: gercek veride 652 kod/alias cakismasi var",
              kod_owner is not None and alias_claimant is not None)

        pool = CanonicalPool(data_dir=data_dir)
        check("is_known_incoming_id('652'): cakisan alias index'e eklenmedigi icin False",
              pool.is_known_incoming_id("652") is False)

        engine = MatchEngine(pool=pool, llm_resolver=None)

        r_id = engine.resolve_by_id("birim_ad", "birim_id", "652").to_dict()
        check("resolve_by_id('652'): resmi kod sahibine (ÇEVRE DÜZENLEME) cozuldu",
              str(r_id["corrected_id"]) == str(kod_owner["kod"]), r_id)
        check("resolve_by_id('652'): adim=kod_id_dogrudan_kod",
              r_id["adim"] == "kod_id_dogrudan_kod", r_id)
        check("resolve_by_id('652'): confidence 0.98",
              abs(r_id["confidence"] - 0.98) < 1e-6, r_id)

        r_text = engine.resolve(FieldSpec(
            field_name="birim_ad", id_field_name="birim_id",
            incoming_text=alias_claimant["ad"], incoming_id="652",
        )).to_dict()
        check("resolve(ID=652 + eski alias-sahibinin adi): yine de dogru birime (alias sahibi) cozuldu",
              str(r_text["corrected_id"]) == str(alias_claimant["kod"]), r_text)
        check("resolve(ID=652 + eski alias-sahibinin adi): emin_degil=False",
              r_text["emin_degil"] is False, r_text)
    finally:
        shutil.rmtree(data_dir)


def test_build_prompt_raw_record_gomuluyor():
    """The raw record only reaches the prompt when it's given."""
    spec_ham_verili = FieldSpec(
        field_name="birim_ad", id_field_name="birim_id",
        incoming_text="TEST BIRIMI", incoming_id="42",
        raw_record={"personelAd": "Ayşe", "personelTckimlikno": "11111111111", "birim_id": 42},
    )
    candidates = [{"uuid": "u1", "ad": "TEST BİRİMİ A.Ş.", "tur": "İDARİ", "kod": "1"}]
    mesajlar = build_prompt("TEST BIRIMI", candidates, spec_ham_verili)
    user_mesaji = mesajlar[-1]["content"]
    check("raw_record JSON olarak promptta var", "11111111111" in user_mesaji, user_mesaji[:300])
    check("ID guvenilmezligi uyarisi promptta var", "FARKLI birimlere karsilik gelebildigi" in user_mesaji)

    spec_ham_verisiz = FieldSpec(
        field_name="birim_ad", id_field_name="birim_id",
        incoming_text="TEST BIRIMI", incoming_id="42",
    )
    mesajlar2 = build_prompt("TEST BIRIMI", candidates, spec_ham_verisiz)
    user_mesaji2 = mesajlar2[-1]["content"]
    check("raw_record yoksa hata vermiyor (mesaj uretildi)", isinstance(user_mesaji2, str) and len(user_mesaji2) > 0)
    check("raw_record yoksa kisisel veri sizmiyor", "11111111111" not in user_mesaji2)
