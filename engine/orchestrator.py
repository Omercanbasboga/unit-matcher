"""The decision tree that resolves one field.

Order, cheapest and most certain first:
  1. ID already known for a unit -> accept, if the text agrees with it
  4. parent unit given -> restrict the search to that branch
  2. synonym table
  2.5 matches learned from human review
  3. exact name match (plus suffix added/removed)
  6. fuzzy search inside the branch; if there's no clear winner, ask the LLM
     to pick from the short list
  7. otherwise leave the field alone and send it to a person

The engine never looks at source-specific field names. Adapters in
scripts/adapters turn a source record into FieldSpecs, so a new source only
needs a new adapter.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from engine.normalize import normalize, turkish_upper
from engine.canonical_lookup import CanonicalPool
from engine.hierarchy import HierarchyEngine, word_similarity, violates_veto
from engine.kadro import KadroEngine
from engine.vector_store import VectorStore

FUZZY_KABUL_ESIGI = 0.55  # below this even the best fuzzy candidate counts as "no candidate"
RAG_EMBED_ESIGI = 0.90  # resolved matches above this go into the vector store
# A known ID still has to agree with the text: if the ID points to unit X but
# the text clearly names something else, that's a conflict for a person.
ISIM_TUTARLILIK_ESIGI = FUZZY_KABUL_ESIGI
# Only near-certain text matches (synonym 0.97, exact 0.95-1.0) may teach the
# pool a new ID. Learned text (0.90-0.93), suffix completion (0.89) and LLM
# picks (<= 0.84) stay below this on purpose.
AUTO_LEARN_ESIGI = 0.95


def _candidate_dicts(fuzzy_sonuc, chosen_uuid: str = None) -> list:
    """Candidates compared in step 6, winner and losers both, so the
    reviewer can see what else was considered."""
    return [
        {
            "uuid": u["uuid"],
            "ad": u["ad"],
            "tur": u.get("tur"),
            "kod": u["kod"],
            "score": round(s, 2),
            "secildi": bool(chosen_uuid) and u["uuid"] == chosen_uuid,
        }
        for s, u in fuzzy_sonuc
    ]


@dataclass
class FieldSpec:
    """One field to resolve, independent of which source it came from."""
    field_name: str            # source field name, for the report
    id_field_name: str         # source ID field name, for the report
    incoming_text: Optional[str]
    incoming_id: Optional[str]
    parent_text: Optional[str] = None      # parent unit text, if the source has one
    unvan: Optional[str] = None
    jobrecord_tipi: Optional[str] = None
    fiili_gorev_metni: Optional[str] = None  # hint for the acting-role exception
    # The whole raw source record. Only passed to the LLM prompt as extra
    # context, never used by the deterministic steps. Raw IDs in it can be
    # inconsistent across sources; that risk is handled before the LLM call.
    raw_record: Optional[dict] = None


@dataclass
class FieldResult:
    field_name: str
    id_field_name: str
    incoming_text: Optional[str]
    incoming_id: Optional[str]
    corrected_text: Optional[str]
    corrected_id: Optional[str]
    changed: bool
    confidence: float
    reason: str
    emin_degil: bool
    # "kod", "llm", "insan" or "bos". Every call site passes it explicitly.
    # It used to fall back to the default here, and every result, including
    # plain synonym hits, ended up labelled "llm".
    layer: str = "llm"
    # The specific step inside that layer, e.g. "kod_senonim" vs
    # "kod_tam_dize_eslesme".
    adim: str = ""
    # Every candidate compared in step 6. Empty for single-answer steps.
    candidates: list = field(default_factory=list)

    def to_dict(self):
        return {
            "field_name": self.field_name,
            "id_field_name": self.id_field_name,
            "incoming_text": self.incoming_text,
            "incoming_id": self.incoming_id,
            "corrected_text": self.corrected_text,
            "corrected_id": self.corrected_id,
            "changed": self.changed,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "emin_degil": self.emin_degil,
            "layer": self.layer,
            "adim": self.adim,
            "candidates": self.candidates,
        }


def _reject(spec: FieldSpec, reason: str, confidence: float = 0.30, candidates: list = None,
            layer: str = "insan", adim: str = "insan_belirsiz") -> FieldResult:
    """Leave the field as it came in and flag it for review.

    layer defaults to "insan". The one exception is when the LLM was actually
    called and its answer wasn't usable; the caller passes layer="llm" then.
    """
    return FieldResult(
        field_name=spec.field_name,
        id_field_name=spec.id_field_name,
        incoming_text=spec.incoming_text,
        incoming_id=spec.incoming_id,
        corrected_text=spec.incoming_text,
        corrected_id=spec.incoming_id,
        changed=False,
        confidence=confidence,
        reason=reason,
        emin_degil=True,
        layer=layer,
        adim=adim,
        candidates=candidates or [],
    )


class MatchEngine:
    """Entry point tying the canonical pool, hierarchy and staffing rules
    together. Build the pool from another data dir to use another
    organisation's reference data.
    """

    def __init__(self, pool: CanonicalPool = None, llm_resolver: Callable = None,
                 vector_store: VectorStore = None):
        self.pool = pool or CanonicalPool()
        self.hz = HierarchyEngine(self.pool)
        self.kadro = KadroEngine(self.pool.data_dir)
        # llm_resolver(normalized_text, candidates, spec) -> dict | None.
        # Without one, an undecided step 6 goes straight to review.
        self.llm_resolver = llm_resolver
        # Off by default so tests don't write into the real data dir.
        self.vector_store = vector_store
        # New IDs learned during a run. Kept in memory and saved in one go by
        # the run script (feedback.bulk_approve_node_ids); opening a DB
        # connection per record was too slow for runs of thousands.
        self.auto_learned_node_ids = []

    def _learn_new_node_id(self, unit: dict, incoming_id: str, spec: FieldSpec, not_: str) -> None:
        """Add the ID to the in-memory pool right away, so later records in the
        same run benefit, and queue it for saving."""
        incoming_id = str(incoming_id)
        node_list = unit.setdefault("node_id", [])
        if incoming_id not in node_list:
            node_list.append(incoming_id)
        self.pool.by_alias_id[incoming_id] = unit
        self.auto_learned_node_ids.append({
            "node_id": incoming_id,
            "kanonik_uuid": unit["uuid"],
            "secilen_ad": unit["ad"],
            "secilen_kod": unit["kod"],
            "record_id": f"otomatik-ogrenme-{incoming_id}",
            "field_name": spec.field_name,
            "not_": not_,
        })

    def _maybe_embed_for_rag(self, rag_text: str, unit: dict, source_layer: str, confidence: float) -> None:
        """Store confident text matches in the vector store. ID-only lookups
        have no text and are skipped."""
        if self.vector_store is None or not rag_text or confidence < RAG_EMBED_ESIGI:
            return
        self.vector_store.upsert(rag_text, org_unit=unit, layer=source_layer, confidence=confidence)

    def _apply_kadro_and_finish(self, spec: FieldSpec, unit: dict, base_confidence: float, reason: str,
                                 candidates: list = None, source_layer: str = "kod", adim: str = "",
                                 rag_text: str = None) -> FieldResult:
        """Run the staffing check on the unit we found and build the result.

        A forbidden staff-type/unit combination turns the match into a
        rejection. When the rules give a specific ID for this staff type,
        that ID replaces the unit code.
        """
        kategori = self.kadro.kadro_kategorisi(spec.jobrecord_tipi, spec.unvan)
        kadro_sonuc = None
        final_id = unit["kod"]
        confidence = base_confidence
        full_reason = reason

        if kategori != "BİLİNMİYOR":
            kadro_sonuc = self.kadro.check(unit["ad"], kategori, spec.fiili_gorev_metni)
            if kadro_sonuc["kural_bulundu"] and kadro_sonuc["izinli"] is False:
                return _reject(
                    spec,
                    reason=full_reason + " | " + kadro_sonuc["reason"],
                    confidence=0.35,
                    adim="insan_kadro_yasakli_kombinasyon",
                )
            if kadro_sonuc["kural_bulundu"] and kadro_sonuc["izinli_id"]:
                final_id = kadro_sonuc["izinli_id"]
                full_reason += " | " + kadro_sonuc["reason"]
                if kadro_sonuc["bypass_uygulandi"]:
                    full_reason += " (fiili gorev istisnasi uygulandi)"

        # turkish_upper on both sides. With plain .upper(), "Bilgi İşlem" in
        # the source never equalled the canonical name and every untouched
        # field was reported as changed.
        changed = (
            turkish_upper((spec.incoming_text or "").strip()) != turkish_upper(unit["ad"].strip())
            or str(spec.incoming_id or "").strip() != str(final_id).strip()
        )

        self._maybe_embed_for_rag(rag_text, unit, source_layer, confidence)

        return FieldResult(
            field_name=spec.field_name,
            id_field_name=spec.id_field_name,
            incoming_text=spec.incoming_text,
            incoming_id=spec.incoming_id,
            corrected_text=unit["ad"],
            corrected_id=final_id,
            changed=changed,
            confidence=confidence,
            reason=full_reason,
            emin_degil=False,
            layer=source_layer,
            adim=adim,
            candidates=candidates or [],
        )

    def resolve_by_id(self, field_name: str, id_field_name: str, incoming_id: str,
                       unvan: str = None, jobrecord_tipi: str = None) -> FieldResult:
        """For sources that only send an ID and no text.

        Nothing to match here, only to verify: if the ID exists (directly, as
        a known alias, or learned from review) we fill in the official name.
        An unknown ID goes to review; it's never trusted just because it
        looks like an ID.
        """
        spec = FieldSpec(field_name=field_name, id_field_name=id_field_name,
                          incoming_text=None, incoming_id=incoming_id,
                          unvan=unvan, jobrecord_tipi=jobrecord_tipi)
        if not incoming_id:
            return FieldResult(
                field_name=field_name, id_field_name=id_field_name,
                incoming_text=None, incoming_id=incoming_id,
                corrected_text=None, corrected_id=incoming_id,
                changed=False, confidence=1.0,
                reason="ID alani bos - islenecek veri yok (bu normal, orn. bolum/abd atanmamis olabilir).",
                emin_degil=False,
                layer="bos",
                adim="bos_alan_atlandi",
            )

        unit = self.pool.units.get(str(incoming_id))
        if unit is not None:
            return self._apply_kadro_and_finish(
                spec, unit, 0.98,
                "Kaynakta dogrudan kanonik UUID/kod verilmis ve kanonik havuzda "
                "dogrulandi (halusinasyon guvenlik agi gecti).",
                adim="kod_id_dogrudan_uuid",
            )

        # Sources mostly send short numeric codes, not uuids, so the uuid
        # lookup above almost never hits. The unit's own code is as strong as
        # its uuid and is checked before aliases from other systems.
        kod_unit = self.pool.by_kod.get(str(incoming_id))
        if kod_unit is not None:
            return self._apply_kadro_and_finish(
                spec, kod_unit, 0.98,
                "Kaynakta dogrudan kanonik KOD verilmis ve kanonik havuzda "
                "dogrulandi (halusinasyon guvenlik agi gecti).",
                adim="kod_id_dogrudan_kod",
            )

        alias_unit = self.pool.by_alias_id.get(str(incoming_id))
        if alias_unit is not None:
            return self._apply_kadro_and_finish(
                spec, alias_unit, 0.97,
                f"'{incoming_id}' kanonik uuid degil ama '{alias_unit['ad']}' biriminin "
                "baska bir sistemdeki bilinen ID'si (node_id crosswalk'u ile "
                "dogrulandi, halusinasyon guvenlik agi gecti).",
                adim="kod_id_crosswalk_alias",
            )

        # older review channel, still read for compatibility
        learned = self.pool.match_learned_id(incoming_id)
        if learned and learned["org_unit"]:
            return self._apply_kadro_and_finish(
                spec, learned["org_unit"], learned["confidence"],
                f"'{incoming_id}' kanonik havuzda yoktu, ama daha once bir "
                f"insan bu ID'yi '{learned['org_unit']['ad']}' ile eslestirmisti "
                f"(HITL geri besleme, onay_sayisi={learned['onay_sayisi']}).",
                adim="kod_id_hitl_ogrenilmis",
            )
        return _reject(
            spec,
            f"'{incoming_id}' kanonik havuzda bulunamadi - kaynak sistemdeki bu ID "
            "gecersiz/eski olabilir ya da kanonik havuz bu kaydi icermiyor. "
            "ID'ye korukorune guvenilmedi.",
            confidence=0.30,
            adim="insan_id_kanonik_havuzda_yok",
        )

    def resolve(self, spec: FieldSpec) -> FieldResult:
        """Run the decision tree for one field."""
        if not spec.incoming_text or not spec.incoming_text.strip():
            return _reject(
                spec, "Girdi metni bos/None - islenecek veri yok.", confidence=0.30,
                layer="bos", adim="bos_girdi_metni",
            )

        norm_info = normalize(spec.incoming_text, self.pool.data_dir)
        norm_text = norm_info["normalized"]

        # Step 1: the ID is a unit's own code, or a known alias of one.
        # Raw uuids don't count here: a uuid sitting in a text record is only
        # a tie-breaker in step 3. Unknown IDs fall through to step 2+, and if
        # a near-certain text match is found there, the ID is learned.
        # If the ID and the text disagree, the ID is ignored and the text
        # steps still run; the conflict is only mentioned if they fail too.
        if spec.incoming_id is not None:
            id_key = str(spec.incoming_id)
            id_unit = self.pool.by_kod.get(id_key)
            id_kod_dogrudan = id_unit is not None
            if id_unit is None:
                id_unit = self.pool.by_alias_id.get(id_key)
            if id_unit is not None:
                benzerlik = word_similarity(turkish_upper(id_unit["ad"].strip()), norm_text)
                # word_similarity doesn't know about veto pairs: "X DAİRE
                # BAŞKANLIĞI" and "X MÜDÜRLÜĞÜ" share most words but are never
                # the same unit, so the veto is checked here as well.
                tutarli = benzerlik >= ISIM_TUTARLILIK_ESIGI and not violates_veto(
                    turkish_upper(id_unit["ad"].strip()), norm_text
                )
                tutarlilik_kanit_metni = "gelen metin bu birimin adiyla tutarli"

                # Abbreviations look nothing like the official name
                # ("BAP Koordinatörlüğü" vs "BİLİMSEL ARAŞTIRMA PROJELERİ
                # KOORDİNATÖRLÜĞÜ") but still mean that unit. If the text is a
                # synonym or learned text for the *same* unit, accept it.
                # Without this, the first record of a run taught the ID via
                # the synonym table and every later record with that ID was
                # flagged as a conflict. A synonym pointing at a different
                # unit is still a real conflict.
                if not tutarli:
                    syn = self.pool.match_synonym(norm_text)
                    if syn and syn.get("org_unit") and syn["org_unit"]["uuid"] == id_unit["uuid"]:
                        tutarli = True
                        tutarlilik_kanit_metni = (
                            f"gelen metin senonim tablosunda AYNI birime eslendigi icin tutarli "
                            f"sayildi (dogrudan resmi-ad benzerligi={benzerlik:.2f} dusuk olsa bile)"
                        )
                    else:
                        ogrenilmis = self.pool.match_learned(norm_text)
                        if ogrenilmis and ogrenilmis.get("org_unit") and ogrenilmis["org_unit"]["uuid"] == id_unit["uuid"]:
                            tutarli = True
                            tutarlilik_kanit_metni = (
                                f"gelen metin HITL-ogrenilmis metin tablosunda AYNI birime eslendigi "
                                f"icin tutarli sayildi (dogrudan resmi-ad benzerligi={benzerlik:.2f} "
                                f"dusuk olsa bile)"
                            )

                if not tutarli:
                    id_celiski_notu = (
                        f"'{spec.incoming_id}' ID'si bilinen crosswalk'ta '{id_unit['ad']}' "
                        f"birimine karsilik geliyordu ama gelen metin ('{spec.incoming_text}') "
                        f"bu birimin adiyla tutarsizdi (benzerlik={benzerlik:.2f} < "
                        f"{ISIM_TUTARLILIK_ESIGI}) - ID yok sayilip metin uzerinden devam edildi."
                    )
                else:
                    id_celiski_notu = None
                    kaynak_aciklama = (
                        "kaynakta dogrudan kanonik kod verilmis" if id_kod_dogrudan
                        else "bilinen crosswalk'ta bulundu"
                    )
                    return self._apply_kadro_and_finish(
                        spec, id_unit, 0.97,
                        f"'{spec.incoming_id}' ID'si {kaynak_aciklama} - '{id_unit['ad']}' birimine "
                        f"karsilik geliyor ve {tutarlilik_kanit_metni} (ID-oncelikli cozum, "
                        "halusinasyon guvenlik agi gecti).",
                        source_layer="kod",
                        adim="kod_id_dogrudan_kod" if id_kod_dogrudan else "kod_id_crosswalk_oncelikli",
                        rag_text=norm_text,
                    )
            else:
                id_celiski_notu = None
        else:
            id_celiski_notu = None

        # Step 4 runs before 2/3 so they can be checked against the branch.
        parent_unit = None
        candidate_uuids = None
        if spec.parent_text:
            parent_unit = self.hz.resolve_parent_unit(spec.parent_text)
            if parent_unit:
                candidate_uuids = self.hz.candidate_pool_for_parent(parent_unit)

        # Step 2: synonym table
        syn = self.pool.match_synonym(norm_text)
        if syn and syn["org_unit"]:
            reason = syn["reason"]
            if candidate_uuids is not None and syn["org_unit"]["uuid"] not in candidate_uuids:
                # synonym says one branch, the parent field says another
                return _reject(
                    spec,
                    "Senonim eslesmesi bulundu ama ust-birim baglamiyla celisiyor "
                    "(capraz branch celiskisi).",
                    confidence=0.40,
                    adim="insan_senonim_baglam_celiskisi",
                )
            if spec.incoming_id is not None and syn["confidence"] >= AUTO_LEARN_ESIGI:
                self._learn_new_node_id(
                    syn["org_unit"], spec.incoming_id, spec,
                    not_=f"Senonim eslesmesi (confidence={syn['confidence']:.2f}) ile "
                         "otomatik ogrenildi - insan onayi beklenmedi (cok yuksek guven).",
                )
            return self._apply_kadro_and_finish(spec, syn["org_unit"], syn["confidence"], reason,
                                                 source_layer="kod", adim="kod_senonim", rag_text=norm_text)

        # Step 2.5: text a reviewer already resolved once. Checked this early
        # so the same dirty text doesn't land in the review queue again.
        learned = self.pool.match_learned(norm_text)
        if learned and learned["org_unit"]:
            if candidate_uuids is not None and learned["org_unit"]["uuid"] not in candidate_uuids:
                return _reject(
                    spec,
                    "Ogrenilmis (insan onayli) eslesme bulundu ama ust-birim baglamiyla "
                    "celisiyor (capraz branch celiskisi).",
                    confidence=0.40,
                    adim="insan_ogrenilmis_baglam_celiskisi",
                )
            return self._apply_kadro_and_finish(
                spec, learned["org_unit"], learned["confidence"], learned["reason"],
                source_layer="kod", adim="kod_hitl_ogrenilmis_metin", rag_text=norm_text,
            )

        # Step 3: exact name
        exact = self.pool.match_exact(norm_text, candidate_uuids)

        # Some trees split one name into three units by staff type (academic,
        # admin, other). If the name alone is ambiguous, two independent
        # signals can still settle it: the record's own ID matching exactly
        # one candidate, or the person's staff type matching exactly one.
        narrow_reason = None
        if exact["status"] == "belirsiz":
            coklu_adaylar = exact["matches"]
            if spec.incoming_id:
                id_eslesen = [
                    m for m in coklu_adaylar
                    if str(m.get("kod")) == str(spec.incoming_id) or m.get("uuid") == str(spec.incoming_id)
                ]
                if len(id_eslesen) == 1:
                    exact = {"status": "tek_eslesme", "matches": id_eslesen}
                    narrow_reason = (
                        "Ayni isimde birden fazla kanonik kayit vardi; gelen ID alani "
                        f"({spec.incoming_id}) bu adaylardan sadece biriyle ortustugu icin "
                        "o aday secildi."
                    )
            if exact["status"] == "belirsiz":
                kategori = self.kadro.kadro_kategorisi(spec.jobrecord_tipi, spec.unvan)
                if kategori != "BİLİNMİYOR":
                    tur_eslesen = [m for m in coklu_adaylar if m.get("tur") == kategori]
                    if len(tur_eslesen) == 1:
                        exact = {"status": "tek_eslesme", "matches": tur_eslesen}
                        narrow_reason = (
                            "Ayni isimde birden fazla kanonik kayit vardi (kadro tipine gore "
                            f"ayrisan agac); personelin kadro tipi ({kategori}) ile tek adaya "
                            "daraltildi."
                        )

        if exact["status"] == "tek_eslesme":
            unit = exact["matches"][0]
            conf = 1.0 if norm_text == turkish_upper(unit["ad"].strip()) else 0.95
            if narrow_reason:
                conf = min(conf, 0.93)
            if spec.incoming_id is not None and conf >= AUTO_LEARN_ESIGI:
                self._learn_new_node_id(
                    unit, spec.incoming_id, spec,
                    not_=f"Kanonik havuzda birebir dize eslesmesi (confidence={conf:.2f}) ile "
                         "otomatik ogrenildi - insan onayi beklenmedi (cok yuksek guven).",
                )
            return self._apply_kadro_and_finish(
                spec, unit, conf,
                narrow_reason or "Normalizasyon sonrasi kanonik havuzda birebir dize eslesmesi bulundu.",
                source_layer="kod", adim="kod_tam_dize_eslesme", rag_text=norm_text,
            )

        # suffix added or removed
        flexible = self.hz.flexible_exact_match(norm_text, candidate_uuids)
        if flexible:
            return self._apply_kadro_and_finish(
                spec, flexible, 0.89,
                "Eksik/fazla yapisal ek (BÖLÜMÜ/MÜDÜRLÜĞÜ vb.) tamamlanarak/"
                "cikarilarak tek adaya ulasildi.",
                source_layer="kod", adim="kod_esnek_ek_eslesme", rag_text=norm_text,
            )

        if exact["status"] == "belirsiz":
            # same name, no way to tell them apart yet; let step 6 / the LLM try
            adaylar = exact["matches"]
        else:
            adaylar = None

        # Step 6: fuzzy search inside the branch
        scope_riskli = candidate_uuids is None  # no branch: searching the whole pool
        arama_havuzu = candidate_uuids
        if adaylar is not None:
            arama_havuzu = [u["uuid"] for u in adaylar]
            scope_riskli = False

        # Keep the unfiltered list for the reviewer. It used to be overwritten
        # by the thresholded one, so every rejected field showed "no
        # candidates" even when there were near misses. Decisions below still
        # use only the filtered list.
        fuzzy_sonuc_ham = self.hz.constrained_fuzzy_search(norm_text, arama_havuzu, top_k=8)
        fuzzy_sonuc = [(s, u) for s, u in fuzzy_sonuc_ham if s >= FUZZY_KABUL_ESIGI]

        if len(fuzzy_sonuc) == 1 or (
            len(fuzzy_sonuc) >= 2 and fuzzy_sonuc[0][0] - fuzzy_sonuc[1][0] >= 0.15
        ):
            best_score, best_unit = fuzzy_sonuc[0]
            conf = 0.84 if not scope_riskli else 0.75
            conf = min(conf, 0.70 + best_score * 0.15)
            return self._apply_kadro_and_finish(
                spec, best_unit, round(conf, 2),
                f"Bulanik/semantik yakinsama ile tek guclu aday bulundu "
                f"(benzerlik={best_score:.2f}, daraltilmis_havuz={not scope_riskli}).",
                candidates=_candidate_dicts(fuzzy_sonuc_ham, best_unit["uuid"]),
                source_layer="kod", adim="kod_fuzzy_tek_aday", rag_text=norm_text,
            )

        # No clear winner: let the LLM pick from the short list.
        if fuzzy_sonuc and self.llm_resolver is not None:
            candidates_payload = [
                {
                    "uuid": u["uuid"], "ad": u["ad"], "tur": u["tur"],
                    "kod": u["kod"], "ust_kod": u.get("ust_kod"), "score": round(s, 2),
                }
                for s, u in fuzzy_sonuc
            ]
            llm_result = self.llm_resolver(norm_text, candidates_payload, spec)
            if llm_result is not None:
                secilen_uuid = llm_result.get("secilen_uuid")
                chosen_unit = self.pool.units.get(secilen_uuid) if secilen_uuid else None
                if chosen_unit is not None:
                    return self._apply_kadro_and_finish(
                        spec, chosen_unit, llm_result.get("confidence", 0.75),
                        "Yerel LLM (Ollama/Qwen3) daraltilmis aday listesi icinden "
                        f"secim yapti. Gerekce: {llm_result.get('reason', '')}",
                        candidates=_candidate_dicts(fuzzy_sonuc_ham, secilen_uuid),
                        source_layer="llm", adim="llm_daraltilmis_aday_secimi", rag_text=norm_text,
                    )
                # secilen_uuid None means either the model said "none of these"
                # (allowed by the schema) or validate_llm_output caught an
                # invented uuid. Either way llm_fallback already wrote the right
                # reason, so pass it through instead of guessing which one.
                return _reject(
                    spec,
                    f"LLM fallback sonucu: {llm_result.get('reason', 'adaylar arasinda uygun secim yapilamadi')}",
                    confidence=min(llm_result.get("confidence", 0.30), 0.49),
                    candidates=_candidate_dicts(fuzzy_sonuc_ham),
                    layer="llm", adim="insan_llm_secim_yapamadi",
                )

        # Step 7: nothing confident enough. Show the near misses with scores;
        # "aday yok" only when the search really found nothing.
        aday_ozeti = ", ".join(
            f"{u['ad']}({u['kod']}, skor={s:.2f})" for s, u in fuzzy_sonuc_ham[:3]
        ) or "aday yok"
        red_sebebi = (
            f"Hicbir katman (senonim/tam eslesme/hiyerarsi/bulanik arama) "
            f"guven esigini (>={FUZZY_KABUL_ESIGI:.2f}) asan tek bir sonuc uretemedi. "
            f"Esik altinda kalan en yakin adaylar: {aday_ozeti}."
        )
        # keep the ID conflict from step 1 visible to the reviewer
        if id_celiski_notu:
            red_sebebi += " Ayrica: " + id_celiski_notu
        return _reject(
            spec,
            red_sebebi,
            confidence=0.30,
            candidates=_candidate_dicts(fuzzy_sonuc_ham),
            adim="insan_guven_esigi_altinda",
        )
