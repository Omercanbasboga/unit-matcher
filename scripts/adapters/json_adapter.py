"""Generic adapter for flat JSON records.

Field names come from a FieldMap, so most sources only need different flags on
the command line instead of a new adapter.

The source's own unit ID is passed along as incoming_id but never trusted on
its own: the same number can mean a different unit in another system. The
engine checks it against the text before using it.
"""
from dataclasses import dataclass
from typing import Optional

from engine.orchestrator import FieldSpec
from engine.schema import build_record_output


@dataclass
class FieldMap:
    id: str = "id"
    text: str = "birim_ad"
    unit_id: Optional[str] = "birim_id"
    parent: Optional[str] = "ust_birim_ad"
    title: Optional[str] = "unvan"
    staff_type: Optional[str] = "kadro_tipi"
    assignment: Optional[str] = "gorev"  # e.g. "Daire Başkanı V." for acting roles
    fallback_text: Optional[str] = "aciklama"  # used when the text field is empty


def _get(raw: dict, key: Optional[str]):
    return raw.get(key) if key else None


def make_json_adapter(fm: FieldMap = None):
    fm = fm or FieldMap()

    def adapt(raw: dict, engine) -> dict:
        text = _get(raw, fm.text) or _get(raw, fm.fallback_text)
        unit_id = _get(raw, fm.unit_id)
        field_results = []

        if text:
            spec = FieldSpec(
                field_name=fm.text if _get(raw, fm.text) else fm.fallback_text,
                id_field_name=fm.unit_id or "",
                incoming_text=text,
                incoming_id=None if unit_id is None else str(unit_id),
                parent_text=_get(raw, fm.parent),
                unvan=_get(raw, fm.title),
                jobrecord_tipi=_get(raw, fm.staff_type),
                fiili_gorev_metni=_get(raw, fm.assignment),
                raw_record=raw,
            )
            field_results.append(engine.resolve(spec))
        elif unit_id is not None:
            # no text at all: all we can do is verify the ID
            field_results.append(engine.resolve_by_id(
                fm.text, fm.unit_id, str(unit_id),
                unvan=_get(raw, fm.title), jobrecord_tipi=_get(raw, fm.staff_type),
            ))

        record_id = str(_get(raw, fm.id) or "bilinmeyen")
        return build_record_output(record_id, field_results, pool=engine.pool)

    return adapt
