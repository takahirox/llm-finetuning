"""Prompt construction shared by dataset generation and evaluation.

Keeping the prompt in one place guarantees the model is trained and evaluated
against identical instructions.
"""

from __future__ import annotations

from .schema import VALID_MODES

SYSTEM_PROMPT = (
    "You are a logistics data extractor. Given a freeform shipment sentence, "
    "extract a single JSON object with exactly these fields: "
    '"origin" (city, string), "destination" (city, string), '
    '"weight_lbs" (integer pounds), "mode" (one of '
    + "/".join(VALID_MODES)
    + '), and "on_time" (boolean). '
    "Respond with only the JSON object and nothing else."
)


def build_user_prompt(text: str) -> str:
    """Build the user turn for a given shipment sentence."""
    return f"Shipment: {text}\nExtract the JSON:"


def build_messages(text: str) -> list[dict[str, str]]:
    """Build the system+user chat messages (no assistant turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]
