from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

SYSTEM_PROMPT = "Journal interpretation assistant. Be conservative. Return JSON."
DEFAULT_MAX_ITEMS = 300

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "inode_nr": {"type": "integer"},
                    "candidate_name": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["inode_nr", "candidate_name", "confidence"],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["events", "notes"],
}


@dataclass(slots=True)
class JournalCandidate:
    inode_nr: int


@dataclass(slots=True)
class InterpretedJournalEvent:
    inode_nr: int
    candidate_name: str
    confidence: float


@dataclass(slots=True)
class JournalInterpretationResult:
    events: list[InterpretedJournalEvent]
    notes: list[str]
    raw_response: str = ""


class JournalInterpreter:
    def __init__(self, api_key: str | None = None, model: str = "gpt-5") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: openai. Install with `pip install openai`."
            ) from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def interpret(
        self,
        items: Iterable[JournalCandidate],
        *,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> JournalInterpretationResult:
        item_list = list(items)
        if len(item_list) > max_items:
            raise ValueError(
                f"Too many journal items: {len(item_list)} (max_items={max_items})."
            )

        payload = {"items": [asdict(x) for x in item_list]}
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "journal_interpretation_result",
                    "strict": True,
                    "schema": JSON_SCHEMA,
                },
            },
        )
        raw_text = response.output_text
        data = _parse_result_json(raw_text)
        return JournalInterpretationResult(
            events=_parse_events(data["events"]),
            notes=_parse_notes(data["notes"]),
            raw_response=raw_text,
        )


def _parse_result_json(raw_text: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI response is not valid JSON: {exc.msg}.") from exc
    if not isinstance(data, dict):
        raise ValueError("AI response must be a JSON object.")
    if set(data.keys()) != {"events", "notes"}:
        raise ValueError("AI response must contain exactly: events, notes.")
    return data


def _parse_events(raw_events: Any) -> list[InterpretedJournalEvent]:
    if not isinstance(raw_events, list):
        raise ValueError("AI response field 'events' must be an array.")
    events: list[InterpretedJournalEvent] = []
    for idx, event in enumerate(raw_events):
        if not isinstance(event, dict):
            raise ValueError(f"AI response events[{idx}] must be an object.")
        required_keys = {"inode_nr", "candidate_name", "confidence"}
        extra = set(event.keys()) - required_keys
        missing = required_keys - set(event.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"AI response events[{idx}] missing field(s): {missing_str}.")
        if extra:
            extra_str = ", ".join(sorted(extra))
            raise ValueError(f"AI response events[{idx}] has unexpected field(s): {extra_str}.")
        inode_nr = event["inode_nr"]
        candidate_name = event["candidate_name"]
        confidence = event["confidence"]
        if not isinstance(inode_nr, int):
            raise ValueError(f"AI response events[{idx}].inode_nr must be an integer.")
        if not isinstance(candidate_name, str):
            raise ValueError(f"AI response events[{idx}].candidate_name must be a string.")
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"AI response events[{idx}].confidence must be a number.")
        events.append(
            InterpretedJournalEvent(
                inode_nr=inode_nr,
                candidate_name=candidate_name,
                confidence=float(confidence),
            )
        )
    return events


def _parse_notes(raw_notes: Any) -> list[str]:
    if not isinstance(raw_notes, list):
        raise ValueError("AI response field 'notes' must be an array.")
    notes: list[str] = []
    for idx, note in enumerate(raw_notes):
        if not isinstance(note, str):
            raise ValueError(f"AI response notes[{idx}] must be a string.")
        notes.append(note)
    return notes
