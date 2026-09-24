"""Behavioral ontology plugins."""

from .persistent import PersistentPlugin
from .reactive_jpy import ReactiveJPYPlugin


def make_plugin(ontology_id: str):
    oid = str(ontology_id).strip().lower()
    if oid == "reactive_jpy":
        return ReactiveJPYPlugin()
    if oid == "persistent":
        return PersistentPlugin()
    raise ValueError(f"unsupported ontology_id: {ontology_id!r}")


__all__ = ["ReactiveJPYPlugin", "PersistentPlugin", "make_plugin"]
