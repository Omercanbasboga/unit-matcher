"""Local LLM fallback (Qwen3 on Ollama).

Only called when the fuzzy step can't settle on one candidate. The model
doesn't need to know the 300+ canonical units: it gets a short closed list
(at most 8) and either picks one or says none fit. It only ever sees uuids
that exist, and validate_llm_output checks the answer anyway.

Talks to Ollama's REST API with urllib, so neither the ollama nor the
requests package is needed.
"""
import json
import os
import sys
import urllib.request
import urllib.error

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

# Typing secilen_uuid as just a string let the model return any string, and
# qwen3:8b did return a uuid that wasn't on the list. Building an enum of the
# candidate uuids per call makes Ollama's constrained decoding rule that out.
# Older Ollama versions may ignore the schema; validate_llm_output still
# catches it.
def build_response_schema(candidates: list) -> dict:
    valid_uuids = [c["uuid"] for c in candidates]
    return {
        "type": "object",
        "properties": {
            "secilen_uuid": {"enum": valid_uuids + [None]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["secilen_uuid", "confidence", "reason"],
    }

VETO_HATIRLATMA = (
    "KESIN KURAL: 'DAİRE BAŞKANLIĞI' ile 'MÜDÜRLÜĞÜ' birbirinin yerine "
    "gecmez. 'FAKÜLTESİ' ile 'BÖLÜMÜ' veya 'ANABİLİM DALI' birbirinin "
    "yerine gecmez. 'KOORDİNATÖRLÜĞÜ' ile 'MERKEZİ' veya 'OFİSİ' birbirinin "
    "yerine gecmez. Bu kelime ciftleri arasinda kelime benzerligi ne kadar "
    "yuksek olursa olsun ASLA birbirine esleme yapma."
)


RULESET_SYSTEM_TEMPLATE = """Sen kurumun ruleset.md dokumaninda tarif edilen kurallara
tam hakim, kurumsal birim eslestirme konusunda uzmanlasmis bir asistansin.
Asagida o kurumun TAM ruleset.md icerigi, arka plan/baglam olarak veriliyor -
hiyerarsiyi, isimlendirme kurallarini ve ozel durumlari daha iyi anlaman icin.

═══════════════════════════════════════════════════════════════
RULESET.MD (TAM ICERIK - SADECE BAGLAM/KURAL REHBERI)
═══════════════════════════════════════════════════════════════
{ruleset_text}
═══════════════════════════════════════════════════════════════

KESIN VE ASLA IHLAL EDILMEYECEK KISIT (yukaridaki belgeden BAGIMSIZ olarak
her zaman gecerlidir): Yukarida gordugun ruleset.md sana SADECE baglam
vermek icindir. Her tek istek mesajinda ayrica sana KAPALI (kisitli) bir
aday listesi verilecek. NIHAI SECIMINI SADECE o mesajdaki kapali listeden
yapmak ZORUNDASIN - ruleset.md icinde gordugun ama o mesajdaki listede
GECMEYEN hicbir birimi/uuid'yi secme, oneri, uydurma. Ruleset.md'yi ezbere
tekrar etmek veya listede olmayan "daha dogru" bir birim onermek YASAKTIR;
bu bir halusinasyon guvenlik agi tarafindan zaten programatik olarak
reddedilecektir. Emin degilsen veya kapali listedeki hicbir aday uygun
degilse secilen_uuid alanini null birak - bu mesru bir cevaptir, ceza degildir.

{veto_hatirlatma}
"""


def build_ruleset_system_prompt(ruleset_text: str) -> str:
    """Embed the full rules doc in the system prompt, built once per resolver.

    This gives the model far more context than the 8-item list, so the
    "pick only from the list" paragraph and the output validation matter
    more here, not less.
    """
    return RULESET_SYSTEM_TEMPLATE.format(ruleset_text=ruleset_text.strip(), veto_hatirlatma=VETO_HATIRLATMA)


def _format_rag_block(rag_examples: list) -> str:
    """Format past confirmed matches for the prompt. They're context only and
    never add candidates."""
    if not rag_examples:
        return ""
    lines = "\n".join(
        f'- "{ex["normalized_text"]}" (benzerlik={ex["score"]:.2f}) -> '
        f'{ex["resmi_isim"]} (kod={ex["kanonik_kod"]}, onaylayan_katman={ex["layer"]}, '
        f'guven={ex["confidence"]})'
        for ex in rag_examples
    )
    return f"""
Gecmiste bu motorun (kod katmani / LLM / insan onayi ile) DAHA ONCE onayladigi,
girdiye ANLAMCA yakin ornekler (RAG - Vektor Database'den getirildi). Bunlar
SADECE referans/baglam icindir, yeni aday EKLEMEZLER - nihai secimini yine
asagidaki KAPALI aday listesinden yapmak ZORUNDASIN:
{lines}
"""


def _format_ham_veri_block(raw_record) -> str:
    """Format the raw source record for the prompt (empty string if none).

    Raw IDs in it can contradict each other across sources, so the prompt
    also tells the model not to treat them as evidence.
    """
    if not raw_record:
        return ""
    try:
        ham_json = json.dumps(raw_record, ensure_ascii=False, default=str)
    except TypeError:
        return ""
    return f"""
Kaynak sistemden gelen TAM ham kayit (ek baglam/referans icindir):
```json
{ham_json}
```
ONEMLI: bu ham kayittaki ID alanlari (isminde "id" gecen alanlar) FARKLI
kaynak sistemlerde AYNI ID'nin FARKLI birimlere karsilik gelebildigi
dogrulanmis bir kurumsal veri sorunu tasiyabilir. Bu ID'lere GUVENEREK ya
da onlari "kanit" sayarak aday secme - kararini SADECE birim adi/hiyerarsi
baglamina gore ver.
"""


def build_prompt(normalized_text: str, candidates: list, spec, system_prompt: str = None,
                  rag_examples: list = None) -> list:
    """Build the chat messages.

    Without a system prompt the veto reminder goes into the user message.
    With one (see build_ruleset_system_prompt) it's already in the system
    message. rag_examples and spec.raw_record each add an optional block.
    """
    candidate_lines = "\n".join(
        f"- uuid={c['uuid']} | ad=\"{c['ad']}\" | tur={c['tur']} | kod={c['kod']}"
        for c in candidates
    )
    context_bits = []
    if getattr(spec, "unvan", None):
        context_bits.append(f"Personelin unvani: {spec.unvan}")
    if getattr(spec, "jobrecord_tipi", None):
        context_bits.append(f"Kadro tipi: {spec.jobrecord_tipi}")
    if getattr(spec, "parent_text", None):
        context_bits.append(f"Bilinen ust birim baglami: {spec.parent_text}")
    context_str = "\n".join(context_bits) if context_bits else "(ek baglam yok)"

    veto_block = "" if system_prompt else f"\n{VETO_HATIRLATMA}\n"
    rag_block = _format_rag_block(rag_examples)
    ham_veri_block = _format_ham_veri_block(getattr(spec, "raw_record", None))

    user_prompt = f"""Sen bir kurumsal birim eslestirme uzmanisin. Asagida KIRLI/normalize
edilmis bir birim adi ve bu birimin olabilecegi KAPALI (sinirli) aday listesi var.
Gorevin: SADECE bu listeden birini secmek, ya da hicbiri uygun degilse
secilen_uuid alanini null birakmak. Listede OLMAYAN bir UUID veya isim
UYDURAMAZSIN.
{veto_block}
Girdi (normalize edilmis) birim metni: "{normalized_text}"

Baglam:
{context_str}
{ham_veri_block}
Aday listesi (SADECE bunlardan biri secilebilir):
{candidate_lines}
{rag_block}
Sadece asagidaki JSON semasina uyan TEK bir JSON nesnesi don, baska hicbir
metin ekleme:
{{"secilen_uuid": "<listeden bir uuid ya da null>", "confidence": <0.0-1.0 arasi sayi>, "reason": "<kisa Turkce gerekce>"}}
"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def call_ollama(messages: list, response_schema: dict, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, num_ctx: int = 8192, timeout: int = 180,
                 think: bool = False) -> str:
    """POST to Ollama's /api/chat and return the model's raw content.

    Raises RuntimeError when Ollama is down, unreachable or too slow; the
    resolver turns that into None so the field just goes to review.

    think=False: Qwen3 runs a hidden reasoning pass by default. In testing,
    one simple question with thinking on blew a 60s timeout on the 8b
    model; with it off it took about a second. Picking from a closed list
    doesn't need it. Models without thinking support ignore the flag.
    """
    payload = {
        "model": model,
        "messages": messages,
        "format": response_schema,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
        "stream": False,
        "think": think,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except OSError as e:
        # covers URLError/HTTPError and TimeoutError (an OSError since 3.10)
        raise RuntimeError(f"Ollama'ya baglanilamadi/zaman asimi ({OLLAMA_URL}): {e}") from e
    return body.get("message", {}).get("content", "")


def validate_llm_output(raw_content: str, candidates: list) -> dict:
    """Whatever the model says, check that the uuid is really on the list.
    Invented uuids or broken JSON become a rejection."""
    valid_uuids = {c["uuid"] for c in candidates}

    try:
        parsed = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return {"secilen_uuid": None, "confidence": 0.30,
                "reason": "LLM cikisi gecerli JSON degildi; guvenlik agi tarafindan reddedildi."}

    if not isinstance(parsed, dict) or "secilen_uuid" not in parsed:
        return {"secilen_uuid": None, "confidence": 0.30,
                "reason": "LLM cikisi beklenen semaya uymuyordu; reddedildi."}

    uuid = parsed.get("secilen_uuid")
    confidence = parsed.get("confidence", 0.5)
    reason = parsed.get("reason", "")

    if uuid is not None and uuid not in valid_uuids:
        return {
            "secilen_uuid": None,
            "confidence": 0.30,
            "reason": (
                f"LLM sunulan kapali listenin DISINDA bir uuid ({uuid}) uretti - "
                "bu bir ID halusinasyonudur. "
                "Guvenlik agi tarafindan otomatik reddedildi."
            ),
        }

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {"secilen_uuid": uuid, "confidence": confidence, "reason": str(reason)}


def make_llm_resolver(model: str = DEFAULT_MODEL, temperature: float = 0.0, num_ctx: int = 8192,
                       verbose: bool = True, ruleset_text: str = None, timeout: int = 180,
                       think: bool = False, vector_store=None, rag_top_k: int = 3,
                       rag_min_score: float = 0.60):
    """Build the callable passed as MatchEngine(llm_resolver=...).

    Returns None instead of raising when Ollama isn't reachable, so a run
    never crashes because of it.

    vector_store: query it first and add similar past matches to the prompt.
    This happens inside the closure, so the resolver keeps the same
    (text, candidates, spec) signature the fake resolvers in tests use.

    ruleset_text: the full rules doc (~21K tokens) as a system prompt. Give
    num_ctx enough room for it or Ollama silently truncates the context.
    """
    system_prompt = build_ruleset_system_prompt(ruleset_text) if ruleset_text else None

    def _resolver(normalized_text: str, candidates: list, spec):
        if not candidates:
            return None
        rag_examples = None
        if vector_store is not None:
            rag_examples = vector_store.query(normalized_text, top_k=rag_top_k, min_score=rag_min_score)
        messages = build_prompt(normalized_text, candidates, spec, system_prompt=system_prompt,
                                 rag_examples=rag_examples)
        schema = build_response_schema(candidates)
        try:
            raw = call_ollama(messages, schema, model=model, temperature=temperature, num_ctx=num_ctx,
                               timeout=timeout, think=think)
        except RuntimeError as e:
            if verbose:
                print(f"[llm_fallback] UYARI - Ollama'ya erisilemedi: {e}", file=sys.stderr)
            return None
        result = validate_llm_output(raw, candidates)
        if verbose and result["secilen_uuid"] is None and "reddedildi" in result["reason"]:
            print(f"[llm_fallback] UYARI - '{normalized_text}' icin LLM cikisi reddedildi: {result['reason']}",
                  file=sys.stderr)
        return result

    return _resolver


if __name__ == "__main__":
    # Runs without Ollama: prompt building and output validation only.
    fake_candidates = [
        {"uuid": "5d0c2a8e-9b41-4f6e-8a37-1c2e4b6f9a01", "ad": "YABANCI DİLLER YÜKSEKOKULU", "tur": "AKADEMİK", "kod": "156", "ust_kod": "100"},
        {"uuid": "8e3f1b72-46d0-4c19-b5aa-7d2c9e0f3b18", "ad": "YABANCI DİLLER", "tur": "AKADEMİK", "kod": "241", "ust_kod": "156"},
    ]

    class _FakeSpec:
        unvan = "Öğretim Görevlisi"
        jobrecord_tipi = "Akademik"
        parent_text = None

    prompt = build_prompt("YABANCI DİLLER KOORDİNASYON", fake_candidates, _FakeSpec())
    print("--- prompt (first 400 chars) ---")
    print(prompt[0]["content"][:400], "...\n")

    print("--- valid uuid ---")
    ok = validate_llm_output(
        json.dumps({"secilen_uuid": fake_candidates[0]["uuid"], "confidence": 0.8, "reason": "test"}),
        fake_candidates,
    )
    print(ok)

    print("--- invented uuid ---")
    bad = validate_llm_output(
        json.dumps({"secilen_uuid": "00000000-0000-0000-0000-000000000000", "confidence": 0.9, "reason": "uydurma"}),
        fake_candidates,
    )
    print(bad)

    print("--- broken JSON ---")
    broken = validate_llm_output("bu json degil, sohbet metni", fake_candidates)
    print(broken)

    print("\n--- with RAG examples ---")
    fake_rag_examples = [
        {"normalized_text": "YABANCI DİLLER KOORDİNASYON BİRİMİ", "score": 0.82,
         "resmi_isim": "YABANCI DİLLER YÜKSEKOKULU", "kanonik_kod": "156",
         "layer": "insan", "confidence": 0.99},
    ]
    prompt_with_rag = build_prompt("YABANCI DİLLER KOORDİNASYON", fake_candidates, _FakeSpec(),
                                    rag_examples=fake_rag_examples)
    print(prompt_with_rag[0]["content"])
