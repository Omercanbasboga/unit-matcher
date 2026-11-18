"""Optional MongoDB output for analysis: how each field was resolved, and
which other candidates were considered ("found as A, but could be B").

Postgres stays the source of truth for decisions. Losing or corrupting the
Mongo data changes nothing about how the engine behaves.

Same rules as db_sink: pymongo is imported lazily, nothing raises, a failure
only skips this step. Connection settings come from MONGO_* variables.
"""
import os
from urllib.parse import quote_plus

CONNECT_TIMEOUT_MS = 10_000

COZUMLEME_IZI_COLLECTION = "cozumleme_izi"
VARSAYIM_ADAYLARI_COLLECTION = "varsayim_adaylari"


def _connection_uri() -> str:
    """Build a mongodb:// URI from MONGO_* variables."""
    host = os.environ.get("MONGO_HOST", "localhost")
    port = os.environ.get("MONGO_PORT", "27017")
    user = os.environ.get("MONGO_USER", "")
    password = os.environ.get("MONGO_PASSWORD", "")
    db = os.environ.get("MONGO_DB", "unit_matcher")
    if user and password:
        return f"mongodb://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}?authSource=admin"
    return f"mongodb://{host}:{port}/{db}"


def _cozumleme_izi_dokumanlari(kaynak: str, records: list) -> list:
    """One document per field, resolved or not, keyed by
    (kaynak, record_id, field_name)."""
    dokumanlar = []
    for r in records:
        for f in r.get("fields", []):
            dokumanlar.append({
                "kaynak": kaynak,
                "record_id": r["record_id"],
                "field_name": f.get("field_name"),
                "layer": f.get("layer"),
                "adim": f.get("adim"),
                "reason": f.get("reason"),
                "confidence": f.get("confidence"),
                "corrected_text": f.get("corrected_text"),
                "corrected_id": f.get("corrected_id"),
                "emin_degil": f.get("emin_degil"),
            })
    return dokumanlar


def _varsayim_adaylari_dokumanlari(kaynak: str, records: list) -> list:
    """One document per field that had candidates to compare. Synonym and
    exact matches have none and are skipped."""
    dokumanlar = []
    for r in records:
        for f in r.get("fields", []):
            adaylar = f.get("candidates") or []
            if not adaylar:
                continue
            dokumanlar.append({
                "kaynak": kaynak,
                "record_id": r["record_id"],
                "field_name": f.get("field_name"),
                "secilen_id": f.get("corrected_id"),
                "secilen_ad": f.get("corrected_text"),
                "adaylar": adaylar,
            })
    return dokumanlar


def _upsert_toplu(collection, dokumanlar: list) -> int:
    """Upsert all documents in a single bulk_write."""
    from pymongo import UpdateOne

    if not dokumanlar:
        return 0
    islemler = [
        UpdateOne(
            {"kaynak": d["kaynak"], "record_id": d["record_id"], "field_name": d["field_name"]},
            {"$set": d},
            upsert=True,
        )
        for d in dokumanlar
    ]
    collection.bulk_write(islemler, ordered=False)
    return len(islemler)


def write_cozumleme_ve_varsayimlar(
    kaynak: str, records: list, conn_uri: str = None, tam_calistirma: bool = True,
) -> dict:
    """Write the resolution trail and candidate lists for one run.

    With tam_calistirma=True (the default) documents for records that are no
    longer in this source are deleted. Pass False for partial runs
    (--limit), otherwise valid records that just weren't processed this time
    would lose their trail too.

    The two collections are reported separately. With a single flag, a
    failure in one made it look as if nothing had been written.
    Never raises.
    """
    if not records:
        bos = {"basarili": True, "yazilan_kayit": 0, "silinen_bayat_kayit": 0}
        return {"basarili": True, "cozumleme_izi": bos, "varsayim_adaylari": dict(bos)}

    try:
        import pymongo
    except ImportError:
        basarisiz = {
            "basarili": False,
            "sebep": "pymongo kurulu degil. 'pip install -r requirements.txt' ile kurup tekrar deneyin.",
        }
        return {"basarili": False, "cozumleme_izi": basarisiz, "varsayim_adaylari": dict(basarisiz)}

    uri = conn_uri or _connection_uri()
    host = os.environ.get("MONGO_HOST", "?")
    port = os.environ.get("MONGO_PORT", "?")
    try:
        client = pymongo.MongoClient(
            uri, serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS, connectTimeoutMS=CONNECT_TIMEOUT_MS,
        )
        db = client[os.environ.get("MONGO_DB", "unit_matcher")]
    except Exception as e:
        basarisiz = {"basarili": False, "sebep": f"MongoDB'ye baglanilamadi ({host}:{port}): {e}"}
        return {"basarili": False, "cozumleme_izi": basarisiz, "varsayim_adaylari": dict(basarisiz)}

    # Without cleanup, stale candidate lists stayed forever: a field that was
    # ambiguous once and later resolved cleanly kept showing old candidates.
    gecerli_record_idler = list({r["record_id"] for r in records})

    def _koleksiyona_yaz(collection_name: str, dokumanlar: list) -> dict:
        """Upsert, then (for full runs) delete stale documents. Separate
        try blocks: a BulkWriteError in the upsert used to skip the cleanup
        silently."""
        col = db[collection_name]
        yazma_hata = None
        yazilan = 0
        try:
            col.create_index([("kaynak", 1), ("record_id", 1), ("field_name", 1)], unique=True)
            yazilan = _upsert_toplu(col, dokumanlar)
        except Exception as e:
            yazma_hata = f"MongoDB'ye yazilamadi ({host}:{port}): {e}"

        silinen = 0
        silme_hata = None
        if tam_calistirma:
            try:
                silme_sonucu = col.delete_many(
                    {"kaynak": kaynak, "record_id": {"$nin": gecerli_record_idler}}
                )
                silinen = silme_sonucu.deleted_count
            except Exception as e:
                silme_hata = f"Bayat kayitlar silinemedi ({host}:{port}): {e}"

        if yazma_hata or silme_hata:
            return {
                "basarili": False,
                "sebep": " | ".join(s for s in (yazma_hata, silme_hata) if s),
                "yazilan_kayit": yazilan,
                "silinen_bayat_kayit": silinen,
            }
        return {"basarili": True, "yazilan_kayit": yazilan, "silinen_bayat_kayit": silinen}

    try:
        izi_sonuc = _koleksiyona_yaz(COZUMLEME_IZI_COLLECTION, _cozumleme_izi_dokumanlari(kaynak, records))
        varsayim_sonuc = _koleksiyona_yaz(VARSAYIM_ADAYLARI_COLLECTION, _varsayim_adaylari_dokumanlari(kaynak, records))
    finally:
        client.close()

    return {
        "basarili": izi_sonuc["basarili"] and varsayim_sonuc["basarili"],
        "cozumleme_izi": izi_sonuc,
        "varsayim_adaylari": varsayim_sonuc,
    }
