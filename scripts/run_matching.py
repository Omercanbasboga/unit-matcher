"""Run the matcher over a JSON file of records.

  python scripts/run_matching.py --input examples/gorevler.json --output cikti/gorevler.json --no-llm

Writes <output>_eslesen.json (resolved) and <output>_human_review.json (needs a
person); a record is never in both. IDs the engine learned during the run are
saved to the canonical data. Postgres and MongoDB writes are attempted unless
--no-db is given; if they aren't reachable the run just prints a warning.

Field names default to the sample file's; override them with --text-field,
--id-field, --parent-field and so on.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_dotenv  # noqa: E402


def _short(msg: str) -> str:
    return msg.splitlines()[0][:160] if msg else ""


def load_records(input_path: Path) -> list:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list: {input_path}")
    return data


def write_outputs(out_path: Path, results: list) -> dict:
    eslesen = [r for r in results if not r["needs_human_review"]]
    human_review = [r for r in results if r["needs_human_review"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    eslesen_path = out_path.with_name(out_path.stem + "_eslesen.json")
    review_path = out_path.with_name(out_path.stem + "_human_review.json")

    with open(eslesen_path, "w", encoding="utf-8") as f:
        json.dump(eslesen, f, ensure_ascii=False, indent=2)
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(human_review, f, ensure_ascii=False, indent=2)

    return {
        "eslesen": eslesen, "human_review": human_review,
        "eslesen_path": eslesen_path, "review_path": review_path,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", default=None, help="source name stored with the results (default: input file name)")
    ap.add_argument("--canonical-dir", default=None, help="another organisation's data/ folder")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records")
    ap.add_argument("--no-llm", action="store_true", help="skip the Ollama fallback; undecided fields go to review")
    ap.add_argument("--no-rag", action="store_true", help="don't use the vector store")
    ap.add_argument("--no-db", action="store_true", help="don't try Postgres/MongoDB")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--text-field", default="birim_ad")
    ap.add_argument("--unit-id-field", default="birim_id")
    ap.add_argument("--parent-field", default="ust_birim_ad")
    ap.add_argument("--title-field", default="unvan")
    ap.add_argument("--staff-type-field", default="kadro_tipi")
    ap.add_argument("--assignment-field", default="gorev")
    args = ap.parse_args()

    # before any engine import: llm_fallback reads OLLAMA_* at import time
    load_dotenv(PROJECT_ROOT / ".env")

    from engine.canonical_lookup import CanonicalPool
    from engine.orchestrator import MatchEngine
    from scripts.adapters.json_adapter import FieldMap, make_json_adapter

    adapter_fn = make_json_adapter(FieldMap(
        id=args.id_field, text=args.text_field, unit_id=args.unit_id_field,
        parent=args.parent_field, title=args.title_field, staff_type=args.staff_type_field,
        assignment=args.assignment_field,
    ))
    kaynak = args.source or Path(args.input).stem

    pool = CanonicalPool(data_dir=Path(args.canonical_dir) if args.canonical_dir else None)

    vector_store = None
    if not args.no_rag:
        from engine.vector_store import VectorStore
        vector_store = VectorStore(data_dir=pool.data_dir)

    llm_resolver = None
    if not args.no_llm:
        from engine.llm_fallback import make_llm_resolver, DEFAULT_MODEL, OLLAMA_URL
        print(f"LLM fallback on (model={DEFAULT_MODEL}, {OLLAMA_URL}). If Ollama isn't "
              "reachable those fields go to review.")
        llm_resolver = make_llm_resolver(model=DEFAULT_MODEL, vector_store=vector_store)
    else:
        print("LLM fallback off: undecided fields go straight to review.")

    engine = MatchEngine(pool=pool, llm_resolver=llm_resolver, vector_store=vector_store)

    records = load_records(Path(args.input))
    if args.limit:
        records = records[: args.limit]

    results = [adapter_fn(raw, engine) for raw in records]

    written = write_outputs(Path(args.output), results)
    print(f"{len(results)} records")
    print(f"  resolved     : {len(written['eslesen'])} -> {written['eslesen_path']}")
    print(f"  for review   : {len(written['human_review'])} -> {written['review_path']}")

    if engine.auto_learned_node_ids:
        from engine.feedback import bulk_approve_node_ids
        from engine.orchestrator import AUTO_LEARN_ESIGI

        ogrenme = bulk_approve_node_ids(
            engine.auto_learned_node_ids, data_dir=pool.data_dir,
            reviewer="otomatik-motor(cok-yuksek-guven-metin-eslesme)",
        )
        print(f"  learned IDs  : {ogrenme['ogrenilen_kayit']} (text match >= {AUTO_LEARN_ESIGI}), "
              "canonical_units.json updated")
        if not args.no_db and not ogrenme["db"]["basarili"]:
            print(f"  learned IDs  : Postgres not updated ({_short(ogrenme['db']['sebep'])})")

    if args.no_db:
        return

    from engine.db_sink import write_matched_records, write_human_review_records
    from engine.mongo_sink import write_cozumleme_ve_varsayimlar

    for label, fn, rows in (
        ("eslesen_kayitlar", write_matched_records, written["eslesen"]),
        ("human_review_kayitlari", write_human_review_records, written["human_review"]),
    ):
        sonuc = fn(kaynak, rows)
        if sonuc["basarili"]:
            print(f"  postgres     : {label} <- {sonuc['yazilan_kayit']}")
        else:
            print(f"  postgres     : {label} skipped ({_short(sonuc['sebep'])})")

    sonuc = write_cozumleme_ve_varsayimlar(
        kaynak, written["eslesen"] + written["human_review"], tam_calistirma=(args.limit is None),
    )
    for label in ("cozumleme_izi", "varsayim_adaylari"):
        part = sonuc[label]
        if part["basarili"]:
            print(f"  mongo        : {label} <- {part['yazilan_kayit']}, "
                  f"{part.get('silinen_bayat_kayit', 0)} stale removed")
        else:
            print(f"  mongo        : {label} skipped ({_short(part['sebep'])})")


if __name__ == "__main__":
    main()
