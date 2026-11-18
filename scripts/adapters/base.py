"""Adapters turn a source record into FieldSpecs and a record result.

This is the only place that knows what a source's fields mean. The engine
never does, so supporting a new source means writing a new adapter and
nothing else.

    def adapter(raw_record: dict, engine: MatchEngine) -> dict
        # returns engine.schema.build_record_output(...)
"""
from typing import Callable, Iterable
from engine.orchestrator import MatchEngine


def run_adapter_over_records(records: Iterable[dict], adapter: Callable, engine: MatchEngine) -> list:
    return [adapter(raw, engine) for raw in records]
