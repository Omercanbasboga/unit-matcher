"""Staffing-type check: can a person with this kind of post be placed in this unit?

Academic staff (professors, lecturers, research assistants) must have an
academic home unit, administrative staff an administrative one, and a few units
accept neither. kadro_kurallari.json lists, per unit, which ID to use for each
staff category, or "-" when that category can't be placed there.

Units without a row are let through unchanged. Rejecting data because of a rule
that doesn't exist would be its own kind of made-up answer.
"""
import json

from engine.config import get_data_dir
from engine.normalize import turkish_upper

# titles whose home unit has to be academic
AKADEMIK_UNVANLAR = {
    "PROFESÖR", "PROFESÖR DR.", "PROF.", "PROF. DR.",
    "DOÇENT", "DOÇENT DR.", "DOÇ.", "DOÇ. DR.",
    "DOKTOR ÖĞRETİM ÜYESİ", "DR. ÖĞR. ÜYESİ",
    "ARAŞTIRMA GÖREVLİSİ", "ARAŞ. GÖR.",
    "ÖĞRETİM GÖREVLİSİ", "ÖĞR. GÖR.",
}

IDARI_UNVANLAR = {"MEMUR", "BİLGİSAYAR İŞLETMENİ", "PROGRAMCI", "ŞEF"}

# An academic who is acting head/advisor of an admin unit is a temporary
# assignment, not a placement error. These markers lift the academic
# restriction and the admin ID is used instead.
FIILI_GOREV_IDARI_ANAHTAR_KELIMELER = [
    "VEKİLİ", " V.", "DANIŞMANI", "KOORDİNATÖR", "BAŞKANI V.", "GÖREVLENDİRME",
]


def _load_kadro_kurallari(data_dir=None):
    data_dir = data_dir or get_data_dir()
    with open(data_dir / "kadro_kurallari.json", encoding="utf-8") as f:
        rows = json.load(f)
    return {turkish_upper(r["birim_adi"].strip()): r for r in rows}


class KadroEngine:
    """Staffing rules for one data dir."""

    def __init__(self, data_dir=None):
        self.table = _load_kadro_kurallari(data_dir)

    def kadro_kategorisi(self, jobrecord_tipi: str = None, unvan: str = None) -> str:
        """AKADEMİK / İDARİ / DİĞER PERSONEL / BİLİNMİYOR.

        jobrecord_tipi wins when the source has it; otherwise guess from the
        title. turkish_upper matters here: "Akademik".upper() gives
        "AKADEMIK" with a dotless I, which never equals "AKADEMİK", and every
        comparison below would quietly be False.
        """
        if jobrecord_tipi:
            t = turkish_upper(jobrecord_tipi.strip())
            if t == "AKADEMİK":
                return "AKADEMİK"
            if t == "İDARİ":
                return "İDARİ"
            if "DİĞER" in t:
                return "DİĞER PERSONEL"
        if unvan:
            u = turkish_upper(unvan.strip())
            if any(u.startswith(a) or a in u for a in AKADEMIK_UNVANLAR):
                return "AKADEMİK"
            if any(u.startswith(a) or a in u for a in IDARI_UNVANLAR):
                return "İDARİ"
        return "BİLİNMİYOR"

    def fiili_gorev_idari_mi(self, gorev_metni: str) -> bool:
        """Does the title/assignment text look like an acting admin role?"""
        if not gorev_metni:
            return False
        u = turkish_upper(gorev_metni.strip())
        return any(k in u for k in FIILI_GOREV_IDARI_ANAHTAR_KELIMELER)

    def check(self, resmi_birim_adi: str, kadro_kategorisi: str, fiili_gorev_metni: str = None):
        """Check the unit's row in the staffing table.

        Returns {"kural_bulundu", "izinli" (None when there's no rule),
        "izinli_id", "bypass_uygulandi", "reason"}.
        """
        key = turkish_upper((resmi_birim_adi or "").strip())
        row = self.table.get(key)

        if row is None:
            return {
                "kural_bulundu": False,
                "izinli": None,
                "izinli_id": None,
                "bypass_uygulandi": False,
                "reason": (
                    f"'{resmi_birim_adi}' icin kadro_kurallari'nda satir yok. "
                    "Kisitlama uygulanamadi, veri oldugu gibi birakildi."
                ),
            }

        col_map = {
            "İDARİ": "idari_id",
            "DİĞER PERSONEL": "diger_personel_id",
            "AKADEMİK": "akademik_id",
        }
        col = col_map.get(kadro_kategorisi)
        if not col:
            return {
                "kural_bulundu": True,
                "izinli": None,
                "izinli_id": None,
                "bypass_uygulandi": False,
                "reason": f"Bilinmeyen kadro kategorisi: {kadro_kategorisi}",
            }

        value = row[col]
        gecerli = not value.strip().startswith("-")

        if not gecerli and kadro_kategorisi == "AKADEMİK" and self.fiili_gorev_idari_mi(fiili_gorev_metni):
            # acting admin role: lift the academic restriction, use the admin ID
            idari_value = row["idari_id"]
            return {
                "kural_bulundu": True,
                "izinli": True,
                "izinli_id": idari_value if not idari_value.strip().startswith("-") else None,
                "bypass_uygulandi": True,
                "reason": (
                    "Akademik kadro kisiti ihlal ediliyordu ama fiili gorev "
                    "(idari gorevlendirme) istisnasi tespit edildi; idari ID kullanildi."
                ),
            }

        if not gecerli:
            return {
                "kural_bulundu": True,
                "izinli": False,
                "izinli_id": None,
                "bypass_uygulandi": False,
                "reason": (
                    f"Kadro kurallarina gore '{resmi_birim_adi}' birimine "
                    f"'{kadro_kategorisi}' kadro tipiyle atama GEÇERSİZ (ATANAMAZ)."
                ),
            }

        return {
            "kural_bulundu": True,
            "izinli": True,
            "izinli_id": value,
            "bypass_uygulandi": False,
            "reason": f"Kadro kurallarina gore izinli, ID={value}.",
        }


if __name__ == "__main__":
    ke = KadroEngine()

    print("--- staff category ---")
    print(ke.kadro_kategorisi(jobrecord_tipi="Akademik"))
    print(ke.kadro_kategorisi(unvan="Programcı"))
    print(ke.kadro_kategorisi(jobrecord_tipi="Diğer Personel"))

    print()
    print("--- professor -> IT department ---")
    r = ke.check("BİLGİ İŞLEM DAİRE BAŞKANLIĞI", "AKADEMİK")
    print(r)

    print()
    print("--- same, but acting head ('Daire Başkanı V.') ---")
    r2 = ke.check("BİLGİ İŞLEM DAİRE BAŞKANLIĞI", "AKADEMİK", fiili_gorev_metni="Daire Başkanı V.")
    print(r2)

    print()
    print("--- admin staff, same unit ---")
    r3 = ke.check("BİLGİ İŞLEM DAİRE BAŞKANLIĞI", "İDARİ")
    print(r3)

    print()
    print("--- unit with no rule ---")
    r4 = ke.check("EĞİTİM FAKÜLTESİ", "AKADEMİK")
    print(r4)
