# unit-matcher

Matches messy organisation-unit names to an official unit list. Rules go first, a local LLM only gets the cases the rules can't settle, and whatever is still uncertain goes to a person. Their answers are remembered, so the same mess doesn't come back next time.

It started as a tool I built at work, where staff and assignment records from several systems had to be tied to one official org tree. What's here is the engine and its rules. Everything tied to the original organisation (data, source adapters, integrations) has been taken out, and `data/` is a small made-up university.

## The problem

The same unit shows up in a dozen forms:

```
Bilgi İşlem D. Bşk.            -> BİLGİ İŞLEM DAİRE BAŞKANLIĞI
MÜHENDİSLİK   FAKÜLTESİİİ      -> MÜHENDİSLİK FAKÜLTESİ
BAP Koordinatörlüğü            -> BİLİMSEL ARAŞTIRMA PROJELERİ KOORDİNATÖRLÜĞÜ
Tahakkuk Şb. Müd.              -> TAHAKKUK ŞUBE MÜDÜRLÜĞÜ
```

and the obvious approach, "pick the most similar name", gets the important cases wrong. "Sağlık Kültür ve Spor Müdürlüğü" shares almost every word with "Sağlık Kültür ve Spor Daire Başkanlığı", but a directorate (müdürlük) is never the department head office (daire başkanlığı) above it. On top of that, the IDs that come with the records can't be trusted on their own. The same number can mean a different unit in another system.

## How a field gets resolved

`engine/orchestrator.py` tries the cheapest, most certain steps first and stops at the first confident answer:

1. **Known ID.** If the record's ID is a unit's own code or a known alias, use it, but only if the text agrees. If it doesn't, the ID is ignored and the text steps still run.
2. **Synonyms.** `BAP KOORDİNATÖRLÜĞÜ`, `MAKİNE MÜHENDİSLİĞİ` and other accepted alternative names.
3. **Learned from review.** Text a reviewer already resolved once.
4. **Exact name** after normalisation (Turkish-aware uppercasing, abbreviations, `(632 SKHK)` style budget suffixes, `FAKÜLTESİİİ` typos). A missing or extra `BÖLÜMÜ` is also tried.
5. **Fuzzy search, inside the right branch.** If the record says which parent unit it belongs to, only that subtree is searched. Some suffix pairs can never match each other, however similar the rest of the name is (daire başkanlığı / müdürlük, fakülte / bölüm, koordinatörlük / merkez).
6. **LLM.** When there's no clear winner, Qwen3 on Ollama gets the short list (8 at most) and picks one or says none fit. The response schema is an enum of those uuids, and the answer is checked again afterwards, so it can't invent a unit.
7. **Review.** Otherwise the field is left exactly as it came in and flagged, with the near misses and their scores attached.

After a unit is found, a staffing check runs: an academic title can't be placed in a purely administrative unit unless the record shows an acting role ("Daire Başkanı V.").

Every result says which step decided it (`adim`), with what confidence, and why:

```json
{
  "incoming_text": "Beden Eğitimi Bölümü",
  "corrected_text": "Beden Eğitimi Bölümü",
  "emin_degil": true,
  "adim": "insan_guven_esigi_altinda",
  "candidates": [
    {"ad": "BEDEN EĞİTİMİ BÖLÜMÜ", "kod": "151", "score": 1.0},
    {"ad": "BEDEN EĞİTİMİ BÖLÜMÜ", "kod": "239", "score": 1.0}
  ]
}
```

Two departments with the same name under different faculties. Without a parent unit there's no honest way to choose, so it goes to a person. The same text with `"ust_birim_ad": "Spor Bilimleri Fakültesi"` resolves straight to 239.

## Learning

- When a new ID arrives with a near-certain text match (synonym or exact, confidence ≥ 0.95), the ID is added to that unit's aliases. Later records with the same ID take the fast path.
- Reviewer decisions are logged and promoted to lookup tables once they agree (`engine/feedback.py`). If reviewers later disagree, the earlier promotion is taken back instead of being applied forever.
- Confident matches go into a small vector store (character trigrams, no dependencies). The closest past examples are shown to the LLM as context.

## Running it

Python 3.10+. The engine has no dependencies.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python scripts/run_matching.py --input examples/gorevler.json --output cikti/gorevler.json --no-llm --no-db
```

```
24 records
  resolved     : 15 -> cikti/gorevler_eslesen.json
  for review   : 9 -> cikti/gorevler_human_review.json
  learned IDs  : 3 (text match >= 0.95), canonical_units.json updated
```

The run updates `data/canonical_units.json` with the IDs it learned; `git checkout data/` puts the sample back.

For the LLM step, run Ollama with `qwen3:8b` (or set `OLLAMA_MODEL`) and drop `--no-llm`. If Ollama isn't reachable, those fields go to review and nothing crashes.

Postgres and MongoDB are optional. `docker compose up -d` starts both, `scripts/seed_postgres.py` loads the schema and reference data, and without `--no-db` each run also writes its results there. Mongo only keeps the trail of how each field was resolved and which other candidates were considered. Postgres is the source of truth.

Another organisation's data works the same way: point `UNIT_MATCHER_DATA_DIR` (or `--canonical-dir`) at a folder with the same four JSON files. Different field names in the input only need flags (`--text-field`, `--parent-field`, ...), or a new adapter in `scripts/adapters/`.

## Tests

```bash
pytest
```

48 tests covering the decision tree, ID/text conflicts, the review loop, the vector store, and the database writers failing quietly when there's no database.

## Known limits

- Text typed without Turkish characters ("bilgi islem daire baskanligi") isn't folded, so it ends up in review.
- When two candidates tie exactly, the review message still lists them as "below the threshold" even though both scored 1.0. The candidates are right, the wording isn't.
- The staffing table in the sample has only a handful of rows. Units without a row aren't restricted at all.
- Identifiers and messages are in Turkish because the data is.
