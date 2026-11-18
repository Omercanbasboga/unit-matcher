"""Record-level output: combine a record's field results into one summary.

  overall_confidence  the lowest field confidence; a record is as reliable
                      as its weakest field, so no averaging
  needs_human_review  any field flagged uncertain or below 0.50
  dominant_branch     top-level branch (directly under REKTÖRLÜK) of the most
                      confident resolved field; None if nothing resolved
"""
from engine.canonical_lookup import CanonicalPool
from engine.normalize import turkish_upper


def _find_s1_ancestor(pool: CanonicalPool, unit: dict):
    """Walk up to REKTÖRLÜK and return the ancestor right below it
    (or the unit itself if it already sits there)."""
    if not unit:
        return None
    chain = [unit] + pool.parent_chain(unit["uuid"])
    root = None
    for u in reversed(chain):
        if turkish_upper(u.get("ad", "").strip()) == "REKTÖRLÜK":
            root = u
            break
    if root is None:
        # no REKTÖRLÜK in the chain: return the topmost ancestor we have
        return chain[-1]
    idx = chain.index(root)
    if idx == 0:
        return unit  # the unit is the root itself
    return chain[idx - 1]


def full_hierarchy_chain(pool: CanonicalPool, unit: dict) -> dict:
    """Full path from the root down to the unit, every level included.

    Returns {"zincir": [{"uuid","ad","kod"}, ...], "zincir_metni": "REKTÖRLÜK / ... / unit"}.
    """
    ancestors = pool.parent_chain(unit["uuid"])
    chain = list(reversed(ancestors)) + [unit]
    return {
        "zincir": [{"uuid": u["uuid"], "ad": u["ad"], "kod": u["kod"]} for u in chain],
        "zincir_metni": " / ".join(u["ad"] for u in chain),
    }


def aggregate_fields(fields: list) -> tuple:
    """(overall_confidence, needs_human_review) for a list of field dicts.
    Kept in one place so a re-review recomputes it the same way."""
    if not fields:
        return 0.0, True
    overall_confidence = min(f["confidence"] for f in fields)
    needs_human_review = any(f["emin_degil"] or f["confidence"] < 0.50 for f in fields)
    return overall_confidence, needs_human_review


def build_record_output(record_id: str, field_results: list, pool: CanonicalPool = None) -> dict:
    fields = [fr.to_dict() for fr in field_results]

    if not fields:
        return {
            "record_id": record_id,
            "fields": [],
            "dominant_branch": None,
            "overall_confidence": 0.0,
            "needs_human_review": True,
        }

    overall_confidence, needs_human_review = aggregate_fields(fields)

    dominant_branch = None
    if pool is not None:
        resolved = [
            (f, fr) for f, fr in zip(fields, field_results)
            if not f["emin_degil"] and f["corrected_id"]
        ]
        if resolved:
            best_f, best_fr = max(resolved, key=lambda pair: pair[0]["confidence"])
            unit = pool.by_kod.get(str(best_f["corrected_id"]))
            if unit:
                s1 = _find_s1_ancestor(pool, unit)
                if s1:
                    dominant_branch = f"REKTÖRLÜK / {s1['ad']}"

    return {
        "record_id": record_id,
        "fields": fields,
        "dominant_branch": dominant_branch,
        "overall_confidence": round(overall_confidence, 2),
        "needs_human_review": needs_human_review,
    }
