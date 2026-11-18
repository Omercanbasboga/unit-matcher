"""Trigram vector store and how the engine feeds it."""
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.canonical_lookup import CanonicalPool
from engine.orchestrator import MatchEngine, FieldSpec, RAG_EMBED_ESIGI
from engine.normalize import normalize
from engine.vector_store import VectorStore
from engine.llm_fallback import build_prompt
from engine import feedback


def check(name, condition, detail=""):
    assert condition, f"{name}: {detail}"


CANONICAL_FILES = ["canonical_units.json", "synonyms.json", "kadro_kurallari.json", "abbreviations.json"]


def _fresh_data_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="unit_matcher_test_vs_"))
    for name in CANONICAL_FILES:
        shutil.copy(PROJECT_ROOT / "data" / name, tmp / name)
    return tmp


def _pick_two_real_units(pool: CanonicalPool):
    units = list(pool.units.values())
    return units[0], units[1]


def test_query_on_empty_store_returns_empty():
    tmp = _fresh_data_dir()
    try:
        vs = VectorStore(data_dir=tmp)
        check("Bos depoda sorgu bos liste donduruyor", vs.query("HERHANGI BIR METIN") == [])
        check("Bos depo len()==0", len(vs) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_upsert_and_query_finds_near_variant():
    tmp = _fresh_data_dir()
    try:
        vs = VectorStore(data_dir=tmp)
        vs.upsert(
            "BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ",
            org_unit={"uuid": "u1", "ad": "BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ", "kod": "101"},
            layer="kod", confidence=0.95,
        )
        check("Upsert sonrasi depo len()==1", len(vs) == 1)

        sonuc = vs.query("BİLGİSAYAR MÜH BÖLÜMÜ", top_k=3, min_score=0.5)
        check("Yakin varyant sorgusu en az 1 sonuc donduruyor", len(sonuc) >= 1, sonuc)
        if sonuc:
            check("Donen sonuc dogru kanonik_uuid'yi tasiyor", sonuc[0]["kanonik_uuid"] == "u1", sonuc)
            check("Donen sonuc dogru resmi_isim'i tasiyor",
                  sonuc[0]["resmi_isim"] == "BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ", sonuc)
            check("Donen sonucta score alani var ve 0-1 araliginda", 0.0 <= sonuc[0]["score"] <= 1.0, sonuc)

        alakasiz = vs.query("TAMAMEN ALAKASIZ BASKA BIR KONU", min_score=0.55)
        check("Alakasiz sorgu bos sonuc donduruyor", alakasiz == [], alakasiz)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_exact_same_text_excluded_from_query():
    tmp = _fresh_data_dir()
    try:
        vs = VectorStore(data_dir=tmp)
        metin = "MAKİNA MÜHENDİSLİĞİ BÖLÜMÜ"
        vs.upsert(metin, org_unit={"uuid": "u2", "ad": metin, "kod": "102"}, layer="kod", confidence=0.95)
        sonuc = vs.query(metin, min_score=0.0)
        check("Birebir ayni metin sorgu sonucunda YOK", sonuc == [], sonuc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_upsert_without_uuid_is_noop():
    tmp = _fresh_data_dir()
    try:
        vs = VectorStore(data_dir=tmp)
        vs.upsert("GECERSIZ GIRDI", org_unit={"ad": "isimsiz"}, layer="kod", confidence=0.9)
        check("uuid'siz org_unit ile upsert hicbir sey eklemiyor", len(vs) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_persistence_round_trip():
    tmp = _fresh_data_dir()
    try:
        vs1 = VectorStore(data_dir=tmp)
        vs1.upsert("KALICILIK TEST BİRİMİ", org_unit={"uuid": "u3", "ad": "KALICILIK TEST BİRİMİ", "kod": "999"},
                   layer="insan", confidence=1.0)

        vs2 = VectorStore(data_dir=tmp)
        check("Yeni VectorStore ornegi diskteki kaydi yukluyor", len(vs2) == 1)
        sonuc = vs2.query("KALICILIK TEST BİRİM", min_score=0.5)
        check("Diskten yuklenen kayit sorgulanabiliyor", len(sonuc) == 1 and sonuc[0]["kanonik_uuid"] == "u3", sonuc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_orchestrator_embeds_high_confidence_exact_match():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        vs = VectorStore(data_dir=data_dir)
        engine = MatchEngine(pool=pool, llm_resolver=None, vector_store=vs)

        unit = pool.units[list(pool.units.keys())[0]]
        spec = FieldSpec(field_name="f", id_field_name="fid", incoming_text=unit["ad"], incoming_id=None)
        r = engine.resolve(spec).to_dict()

        check("Tam eslesme emin_degil=False", r["emin_degil"] is False, r)
        check("Tam eslesme confidence RAG esiginin ustunde", r["confidence"] >= RAG_EMBED_ESIGI, r)
        check("Tam eslesme sonrasi Vektor Database'e embed edildi", len(vs) == 1, len(vs))

        check("Embed edilen kayit kod katmanini isaretliyor",
              any(e["layer"] == "kod" and e["kanonik_uuid"] == unit["uuid"] for e in vs._entries.values()),
              vs._entries)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_orchestrator_does_not_embed_without_vector_store():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        engine = MatchEngine(pool=pool, llm_resolver=None)
        unit = pool.units[list(pool.units.keys())[0]]
        spec = FieldSpec(field_name="f", id_field_name="fid", incoming_text=unit["ad"], incoming_id=None)
        r = engine.resolve(spec).to_dict()
        check("vector_store=None iken resolve() hala calisiyor", r["emin_degil"] is False, r)
        check("vector_store=None iken vector_index.json hic yazilmiyor",
              not (data_dir / "vector_index.json").exists())
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_resolve_by_id_never_embeds():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        vs = VectorStore(data_dir=data_dir)
        engine = MatchEngine(pool=pool, llm_resolver=None, vector_store=vs)

        unit = pool.units[list(pool.units.keys())[0]]
        r = engine.resolve_by_id("job_fk_id", "job_fk_id", unit["uuid"]).to_dict()
        check("resolve_by_id yuksek confidence uretiyor", r["confidence"] >= RAG_EMBED_ESIGI, r)
        check("resolve_by_id hicbir zaman Vektor Database'e embed etmiyor", len(vs) == 0, len(vs))
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_build_prompt_includes_rag_block_when_examples_given():
    class _FakeSpec:
        unvan = None
        jobrecord_tipi = None
        parent_text = None

    candidates = [{"uuid": "u1", "ad": "TEST BİRİMİ", "tur": "İDARİ", "kod": "1"}]
    rag_examples = [{"normalized_text": "TEST BİRİMİ VARYANT", "score": 0.77, "resmi_isim": "TEST BİRİMİ",
                      "kanonik_kod": "1", "layer": "insan", "confidence": 1.0}]

    prompt_no_rag = build_prompt("TEST", candidates, _FakeSpec())
    prompt_with_rag = build_prompt("TEST", candidates, _FakeSpec(), rag_examples=rag_examples)

    check("RAG ornegi verilmezse promptta RAG bloğu yok",
          "Gecmiste bu motorun" not in prompt_no_rag[0]["content"])
    check("RAG ornegi verilirse promptta RAG bloğu var",
          "Gecmiste bu motorun" in prompt_with_rag[0]["content"])
    check("RAG bloğu ornegin resmi ismini iceriyor",
          "TEST BİRİMİ" in prompt_with_rag[0]["content"] and "0.77" in prompt_with_rag[0]["content"],
          prompt_with_rag[0]["content"])


def test_promote_embeds_text_decisions_but_not_id_decisions():
    data_dir = _fresh_data_dir()
    try:
        pool = CanonicalPool(data_dir=data_dir)
        unit_a, unit_b = _pick_two_real_units(pool)

        raw_text = "RAG Testi Icin Kirli Birim Adı QWE789"
        norm_text = normalize(raw_text, data_dir)["normalized"]
        feedback.record_decision(
            data_dir, record_id="k-rag-1", field_name="f", karar_tipi=feedback.KARAR_TIPI_METIN,
            secilen_uuid=unit_a["uuid"], secilen_ad=unit_a["ad"], secilen_kod=unit_a["kod"],
            incoming_text=raw_text, normalized_text=norm_text, reviewer="test-reviewer",
        )
        feedback.record_decision(
            data_dir, record_id="k-rag-2", field_name="job_fk_id_cozumu", karar_tipi=feedback.KARAR_TIPI_ID,
            secilen_uuid=unit_b["uuid"], secilen_ad=unit_b["ad"], secilen_kod=unit_b["kod"],
            incoming_id="fake-id-rag-test", reviewer="test-reviewer",
        )

        result = feedback.promote_to_learned_corrections(data_dir)
        check("Iki karar da terfi etti", len(result["terfi_edilen"]) == 2, result)

        vs = VectorStore(data_dir=data_dir)
        check("Sadece METIN bazli karar Vektor Database'e embed edildi", len(vs) == 1, len(vs))
        entries = list(vs._entries.values())
        if entries:
            check("Embed edilen kaydin katmani 'insan'", entries[0]["layer"] == "insan", entries[0])
            check("Embed edilen kaydin uuid'si dogru", entries[0]["kanonik_uuid"] == unit_a["uuid"], entries[0])
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
