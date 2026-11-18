"""A small vector store for past matches, used as RAG context for the LLM.

Every text -> unit match confirmed with confidence >= 0.90 (by the rules or
the LLM) and every positive human decision is stored here. Right before the
LLM step, the closest past matches are pulled out and shown to the model as
examples of how similar text was resolved before.

Exact repeats are already caught earlier (learned_corrections, step 2.5);
the point here is text that is close but not identical.

No dependencies: "embeddings" are character trigram counts compared with
cosine similarity. Swapping in pgvector or a real embedding model would only
change the inside of this class; callers just use upsert() and query().
"""
import json
import math
from collections import Counter
from pathlib import Path

from engine.config import get_data_dir
from engine.normalize import turkish_upper

VECTOR_INDEX_FILE = "vector_index.json"
N_GRAM = 3


def _ngram_vector(text: str, n: int = N_GRAM) -> Counter:
    """Character trigram counts. The padding lets short words and word
    boundaries show up in the grams too."""
    padded = f"  {turkish_upper((text or '').strip())}  "
    if len(padded) < n:
        return Counter([padded])
    return Counter(padded[i:i + n] for i in range(len(padded) - n + 1))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    ortak = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in ortak)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """Trigram vector store persisted to vector_index.json in the data dir.

    One entry per normalised text: the unit it was last confirmed as, which
    layer confirmed it (kod/llm/insan) and with what confidence. A missing
    file just means an empty store.
    """

    def __init__(self, data_dir=None, path=None):
        self.data_dir = Path(data_dir) if data_dir else get_data_dir()
        self.path = Path(path) if path else self.data_dir / VECTOR_INDEX_FILE
        self._entries = self._load()

    def _load(self) -> dict:
        if not self.path.is_file():
            return {}
        content = self.path.read_text(encoding="utf-8").strip()
        return json.loads(content) if content else {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)

    def upsert(self, normalized_text: str, *, org_unit: dict, layer: str,
               confidence: float, kaynak: str = None, record_id: str = None) -> None:
        """Insert or overwrite the entry for this text.

    The latest decision wins. That keeps the store at one entry per
    distinct text instead of one per record. org_unit needs at least
    uuid, ad and kod.
    """
        if not normalized_text or not org_unit or not org_unit.get("uuid"):
            return
        self._entries[normalized_text] = {
            "normalized_text": normalized_text,
            "kanonik_uuid": org_unit["uuid"],
            "resmi_isim": org_unit.get("ad"),
            "kanonik_kod": org_unit.get("kod"),
            "layer": layer,
            "confidence": round(float(confidence), 2),
            "kaynak": kaynak,
            "record_id": record_id,
        }
        self._save()

    def query(self, normalized_text: str, top_k: int = 3, min_score: float = 0.55) -> list:
        """Closest stored matches above min_score, best first. An empty list
    just means there's no useful context."""
        if not normalized_text or not self._entries:
            return []
        q_vec = _ngram_vector(normalized_text)
        scored = []
        for entry in self._entries.values():
            if entry["normalized_text"] == normalized_text:
                continue  # exact repeats are handled by step 2.5
            score = _cosine(q_vec, _ngram_vector(entry["normalized_text"]))
            if score >= min_score:
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        return [{**entry, "score": round(score, 2)} for score, entry in scored[:top_k]]

    def __len__(self):
        return len(self._entries)


if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="vector_store_demo_"))
    vs = VectorStore(data_dir=tmp)

    vs.upsert(
        "BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ",
        org_unit={"uuid": "u1", "ad": "BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ", "kod": "101"},
        layer="kod", confidence=0.95, kaynak="demo",
    )
    vs.upsert(
        "MAKİNA MÜHENDİSLİĞİ BÖLÜMÜ",
        org_unit={"uuid": "u2", "ad": "MAKİNA MÜHENDİSLİĞİ BÖLÜMÜ", "kod": "102"},
        layer="llm", confidence=0.91, kaynak="demo",
    )

    print(f"{len(vs)} entries")
    print("--- query: 'BİLGİSAYAR MÜH. BÖLÜMÜ' ---")
    for r in vs.query("BİLGİSAYAR MÜH. BÖLÜMÜ"):
        print(r)

    print("--- query: 'HİÇ ALAKASIZ BİR METİN' ---")
    print(vs.query("HİÇ ALAKASIZ BİR METİN") or "(nothing, as expected)")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
