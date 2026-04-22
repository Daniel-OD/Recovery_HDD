from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: openai. Install with `pip install openai`."
    ) from exc


SYSTEM_PROMPT = """\
You are a digital forensics assistant analyzing ext4 jbd2 journal mining results.
Your job:
- Interpret journal-derived candidate file name events conservatively.
- Correlate inode numbers, transaction sequences, timestamps, and directory-entry fragments.
- Identify likely file rename/delete/create patterns.
- Never claim certainty when evidence is partial.
- Return structured JSON only.
Rules:
- jbd2 is noisy and incomplete; be conservative.
- Prefer 'candidate' language over certainty.
- Preserve original extensions if present.
- If evidence is weak, return low confidence and explain why.
"""


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
                    "candidate_parent": {"type": "string"},
                    "sequence": {"type": "integer"},
                    "commit_ts": {"type": "integer"},
                    "event_type": {
                        "type": "string",
                        "enum": ["create", "delete", "rename", "link", "unknown"],
                    },
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "inode_nr",
                    "candidate_name",
                    "candidate_parent",
                    "sequence",
                    "commit_ts",
                    "event_type",
                    "confidence",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["events", "notes"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class JournalCandidate:
    inode_nr: int
    raw_name: str | None = None
    raw_parent: str | None = None
    sequence: int | None = None
    commit_ts: int | None = None
    source_block: int | None = None
    dirent_offset: int | None = None
    record_type: str | None = None
    confidence_hint: float | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InterpretedJournalEvent:
    inode_nr: int
    candidate_name: str
    candidate_parent: str
    sequence: int
    commit_ts: int
    event_type: str
    confidence: float
    reason: str


@dataclass(slots=True)
class JournalInterpretationResult:
    events: list[InterpretedJournalEvent]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [asdict(e) for e in self.events],
            "notes": list(self.notes),
        }


class JournalInterpreter:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5",
        confidence_threshold: float = 0.75,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.confidence_threshold = confidence_threshold

    def interpret(
        self,
        candidates: Iterable[JournalCandidate],
        *,
        max_items: int = 300,
    ) -> JournalInterpretationResult:
        items = list(candidates)[:max_items]
        if not items:
            return JournalInterpretationResult(
                events=[],
                notes=["No journal candidates provided."],
            )

        payload = {
            "task": "interpret_ext4_journal_candidates",
            "candidates": [x.to_prompt_dict() for x in items],
            "instructions": {
                "be_conservative": True,
                "preserve_extensions": True,
                "emit_low_confidence_when_ambiguous": True,
            },
        }

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
                }
            },
        )
        raw = json.loads(response.output_text)
        events = [InterpretedJournalEvent(**e) for e in raw["events"]]
        notes = list(raw["notes"])
        return JournalInterpretationResult(events=events, notes=notes)

    def accepted_events(
        self,
        result: JournalInterpretationResult,
    ) -> list[InterpretedJournalEvent]:
        return [e for e in result.events if e.confidence >= self.confidence_threshold]
