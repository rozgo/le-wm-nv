"""Airframe and task contracts shared by data, evaluation and rendering."""
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_tasks(protocol):
    return protocol["control"].get("tasks", [
        {"kind": kind, "heading": "fixed"} for kind in protocol["control"].get("kinds", [])])


def trace_name(label, split, domain_index, kind, heading="fixed", version=1):
    suffix = f"-{heading}" if version >= 2 else ""
    return f"{label}-{split}-{domain_index}-{kind}{suffix}.json"


def require_x_artifact(metadata):
    if metadata.get("airframe") != "x":
        raise ValueError("JEPA Gym uses one X-frame drone. Older artifacts require their archived source.")


def validate_data(data_dir, protocol):
    manifest = json.loads((data_dir/"manifest.json").read_text())
    require_x_artifact(protocol)
    require_x_artifact(manifest)
    if manifest.get("heading_modes", ["fixed"]) != protocol.get("heading_modes", ["fixed"]):
        raise ValueError("Dataset and protocol heading modes differ")
    return manifest
