"""Deterministic synthetic dataset generator for freight-text -> JSON extraction.

Produces realistic but synthetic shipment sentences paired with their gold JSON
extraction, written in MLX-LM chat format (one JSON object per line with a
"messages" list of system/user/assistant turns). Fully seeded and network-free so
the same seed always yields byte-identical output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .schema import ShipmentExtraction

CITIES = [
    "Toronto",
    "Detroit",
    "Chicago",
    "Dallas",
    "Atlanta",
    "Montreal",
    "Vancouver",
    "Memphis",
    "Denver",
    "Seattle",
    "Phoenix",
    "Newark",
    "Columbus",
    "Houston",
    "Boston",
    "Calgary",
    "Winnipeg",
    "Cleveland",
    "Buffalo",
    "Nashville",
]

MODES = ["LTL", "FTL", "Parcel", "Intermodal"]

# Phrasing templates. {o}=origin {d}=destination {w}=weight string {m}=mode phrase
# {ot}=on-time phrase. We keep mode/on-time phrasing varied to force generalization.
TEMPLATES = [
    "{m} load, {w} from {o} to {d}, {ot}.",
    "Shipment from {o} to {d}: {w}, {m}. Carrier {ot}.",
    "{o} -> {d}, {w}, mode {m}, delivery {ot}.",
    "Booked a {m} move ({w}) {o} to {d}; {ot}.",
    "Freight: {o} to {d}, weighing {w}, {m} service, {ot}.",
    "Need {m} for {w} going {o} to {d}. Status: {ot}.",
    "{w} of cargo, {o} origin, {d} destination, {m}, {ot}.",
    "Pickup in {o}, drop in {d}. {w}. {m}. {ot}.",
]

MODE_PHRASES = {
    "LTL": ["LTL", "less-than-truckload", "LTL freight"],
    "FTL": ["FTL", "full truckload", "full-truckload"],
    "Parcel": ["parcel", "small parcel", "Parcel"],
    "Intermodal": ["intermodal", "rail-truck intermodal", "Intermodal"],
}

ON_TIME_PHRASES = {
    True: ["on-time", "delivered on schedule", "arrived on time", "no delays"],
    False: ["delayed", "running late", "behind schedule", "missed the window"],
}

WEIGHT_FORMATS = [
    lambda w: f"{w:,} lbs",
    lambda w: f"{w} lbs",
    lambda w: f"{w} pounds",
    lambda w: f"{w:,}lb",
]


def _make_example(rng: random.Random) -> dict:
    origin = rng.choice(CITIES)
    destination = rng.choice([c for c in CITIES if c != origin])
    weight = rng.randint(50, 44000)
    mode = rng.choice(MODES)
    on_time = rng.random() < 0.55

    weight_str = rng.choice(WEIGHT_FORMATS)(weight)
    mode_phrase = rng.choice(MODE_PHRASES[mode])
    ot_phrase = rng.choice(ON_TIME_PHRASES[on_time])
    template = rng.choice(TEMPLATES)

    text = template.format(o=origin, d=destination, w=weight_str, m=mode_phrase, ot=ot_phrase)

    gold = ShipmentExtraction(
        origin=origin,
        destination=destination,
        weight_lbs=weight,
        mode=mode,
        on_time=on_time,
    )
    return {"text": text, "gold": gold}


def _to_chat_record(text: str, gold: ShipmentExtraction) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text)},
            {"role": "assistant", "content": gold.to_json()},
        ]
    }


def generate_split(n: int, seed: int) -> list[dict]:
    """Generate `n` chat-format records deterministically from `seed`."""
    rng = random.Random(seed)
    seen: set[str] = set()
    records: list[dict] = []
    while len(records) < n:
        ex = _make_example(rng)
        key = ex["text"]
        if key in seen:
            continue
        seen.add(key)
        records.append(_to_chat_record(ex["text"], ex["gold"]))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main(out_dir: str = "data") -> None:
    out = Path(out_dir)
    splits = {
        "train": (600, 1001),
        "valid": (150, 2002),
        "test": (150, 3003),
    }
    for name, (n, seed) in splits.items():
        records = generate_split(n, seed)
        write_jsonl(records, out / f"{name}.jsonl")
        print(f"wrote {len(records)} -> {out / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
