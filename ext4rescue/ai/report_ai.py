from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: openai. Install with `pip install openai`."
    ) from exc


SYSTEM_PROMPT = """\
You are a digital forensics reporting assistant.
Your job:
- Summarize recovery results for a human operator.
- Be precise, concise, and conservative.
- Distinguish between filesystem-aware recovery, journal-assisted recovery, orphan suggestions, and carving.
- Highlight risk areas, corruption, and uncertainty.
- Return structured JSON only.
Rules:
- Do not exaggerate recovery quality.
- If data is incomplete, say so.
- Prefer operationally useful language.
"""


JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "highlights": {
            "type": "array",
            "items": {"type": "string"},
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "operator_notes": {"type": "string"},
    },
    "required": [
        "executive_summary",
        "highlights",
        "warnings",
        "next_steps",
        "operator_notes",
    ],
    "additionalProperties": False,
}


@dataclass(slots=True)
class ReportAISummary:
    executive_summary: str
    highlights: list[str]
    warnings: list[str]
    next_steps: list[str]
    operator_notes: str | list[str]
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReportAI:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def summarize(
        self,
        report_data: dict[str, Any],
        *,
        max_items: int = 300,
    ) -> ReportAISummary:
        for key, value in report_data.items():
            if isinstance(value, list) and len(value) > max_items:
                raise ValueError(
                    f"Input list '{key}' has {len(value)} items (max_items={max_items})."
                )
        payload = {
            "task": "summarize_recovery_report",
            "report_data": report_data,
            "instructions": {
                "focus_on_operator_decisions": True,
                "distinguish_recovery_sources": True,
                "be_conservative": True,
            },
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "report_ai_summary",
                    "strict": True,
                    "schema": JSON_SCHEMA,
                }
            },
        )
        raw = json.loads(response.output_text)
        return ReportAISummary(**raw, raw_response=response.output_text)
