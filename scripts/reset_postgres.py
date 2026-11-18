"""Drop every table from db/schema.sql. Irreversible, so it needs --yes-i-am-sure.

  python scripts/reset_postgres.py --yes-i-am-sure
  python scripts/seed_postgres.py      # rebuild schema and reference data
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.db_sink import _connection_params  # noqa: E402
from scripts._env import load_dotenv  # noqa: E402

# CASCADE makes the order irrelevant; most dependent first for readability
TABLES = [
    "birim_hiyerarsi",
    "vector_index",
    "kanonik_eksik_bildirimleri",
    "learned_id_aliases",
    "learned_corrections",
    "human_decisions_log",
    "human_review_kayitlari",
    "eslesen_kayitlar",
    "kadro_kurallari",
    "abbreviations",
    "synonyms",
    "canonical_units",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes-i-am-sure", action="store_true",
                     help="Onay bayragi - bu verilmeden script hicbir seyi silmez.")
    args = ap.parse_args()

    if not args.yes_i_am_sure:
        print("[reset_postgres] HICBIR SEY YAPILMADI.")
        print("Bu script db/schema.sql'deki TUM tablolari ve icindeki TUM veriyi")
        print("GERI ALINAMAZ sekilde siler. Onaylamak icin --yes-i-am-sure ekleyin:")
        print("  python scripts/reset_postgres.py --yes-i-am-sure")
        return 1

    load_dotenv(PROJECT_ROOT / ".env")

    import psycopg
    conn_params = _connection_params()
    print(f"[reset_postgres] Baglaniliyor: {conn_params['host']}:{conn_params['port']}/"
          f"{conn_params['dbname']} (kullanici={conn_params['user']})")

    with psycopg.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                print(f"[reset_postgres] DROP TABLE {table} (CASCADE)")
        conn.commit()

    print("[reset_postgres] Tamamlandi - tum tablolar silindi.")
    print("Simdi semayi + referans veriyi yeniden kurmak icin:")
    print("  python scripts/seed_postgres.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
