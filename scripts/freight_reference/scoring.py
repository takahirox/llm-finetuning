"""Evaluation harness: score base vs fine-tuned model on freight->JSON extraction.

The scoring functions (parse, field_exact_match, score_predictions) are pure and
network-free so they can be unit-tested. The model-running entrypoint (`main`)
loads MLX models, generates predictions over test.jsonl, scores both base and
base+adapter, and writes results/*.json + comparison.md.
"""

from __future__ import annotations

import json
import re

from .schema import canonicalize_city, canonicalize_mode

FIELDS = ["origin", "destination", "weight_lbs", "mode", "on_time"]

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_prediction(text: str) -> dict | None:
    """Parse a model output string into a JSON dict, or None if it doesn't parse.

    Tolerant of leading/trailing prose: extracts the first {...} block if a bare
    json.loads fails. Returns None for anything that isn't a JSON object.
    """
    if text is None:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _canon_value(field: str, value: object) -> object:
    """Canonicalize a single field value for comparison."""
    if value is None:
        return None
    if field in ("origin", "destination"):
        return canonicalize_city(str(value))
    if field == "mode":
        return canonicalize_mode(str(value))
    if field == "weight_lbs":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if field == "on_time":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "yes"):
                return True
            if low in ("false", "no"):
                return False
        return None
    return value


def field_exact_match(pred: dict | None, gold: dict, field: str) -> bool:
    """True if pred's canonicalized value for `field` equals gold's."""
    if pred is None or field not in pred:
        return False
    return _canon_value(field, pred[field]) == _canon_value(field, gold[field])


def score_predictions(predictions: list[str], golds: list[dict]) -> dict:
    """Score a list of raw prediction strings against gold dicts.

    Returns json_valid_rate, per-field exact-match rates, field_exact_match
    (micro-average over all fields), and overall_accuracy (all-fields-correct rate).
    """
    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must be the same length")
    n = len(golds)
    if n == 0:
        raise ValueError("need at least one example to score")

    valid = 0
    per_field_correct = dict.fromkeys(FIELDS, 0)
    all_correct = 0
    total_field_hits = 0

    for raw, gold in zip(predictions, golds, strict=True):
        parsed = parse_prediction(raw)
        if parsed is not None:
            valid += 1
        row_all = True
        for field in FIELDS:
            ok = field_exact_match(parsed, gold, field)
            if ok:
                per_field_correct[field] += 1
                total_field_hits += 1
            else:
                row_all = False
        if row_all:
            all_correct += 1

    return {
        "n": n,
        "json_valid_rate": valid / n,
        "field_accuracy": {f: per_field_correct[f] / n for f in FIELDS},
        "field_exact_match": total_field_hits / (n * len(FIELDS)),
        "overall_accuracy": all_correct / n,
    }
