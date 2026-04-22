from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_ITEMS = 300
SYSTEM_PROMPT = (
    "You are a digital forensics reporting assistant. "
    "Return JSON only with an executive summary, highlights, warnings, next steps, and operator notes."
)

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "highlights": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "operator_notes": {"type": "array", "items": {"type": "string"}},
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
    operator_notes: list[str]
    raw_response: str = ""


class ReportAI:
    def __init__(self, api_key: str | None = None, model: str = "gpt-5") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: openai. Install with `pip install openai`."
            ) from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def summarize(
        self,
        data: dict[str, Any],
        *,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> ReportAISummary:
        _validate_input(data, max_items=max_items)
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=json.dumps(data, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "report_ai_summary",
                    "strict": True,
                    "schema": JSON_SCHEMA,
                },
            },
        )
        raw_text = response.output_text
        parsed = _parse_result_json(raw_text)
        return ReportAISummary(
            executive_summary=parsed["executive_summary"],
            highlights=parsed["highlights"],
            warnings=parsed["warnings"],
            next_steps=parsed["next_steps"],
            operator_notes=parsed["operator_notes"],
            raw_response=raw_text,
        )


def _validate_input(data: dict[str, Any], *, max_items: int) -> None:
    if not isinstance(data, dict):
        raise ValueError("Expected report input to be a JSON object.")
    for key, value in data.items():
        if isinstance(value, list) and len(value) > max_items:
            raise ValueError(
                f"Input list '{key}' has {len(value)} items (max_items={max_items})."
            )


def _parse_result_json(raw_text: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI response is not valid JSON: {exc.msg}.") from exc
    if not isinstance(data, dict):
        raise ValueError("AI response must be a JSON object.")
    expected_keys = {
        "executive_summary",
        "highlights",
        "warnings",
        "next_steps",
        "operator_notes",
    }
    if set(data.keys()) != expected_keys:
        raise ValueError(
            "AI response must contain exactly: executive_summary, highlights, warnings, next_steps, operator_notes."
        )

    executive_summary = data["executive_summary"]
    if not isinstance(executive_summary, str):
        raise ValueError("AI response field 'executive_summary' must be a string.")
    return {
        "executive_summary": executive_summary,
        "highlights": _parse_string_list(data["highlights"], "highlights"),
        "warnings": _parse_string_list(data["warnings"], "warnings"),
        "next_steps": _parse_string_list(data["next_steps"], "next_steps"),
        "operator_notes": _parse_string_list(data["operator_notes"], "operator_notes"),
    }


def _parse_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"AI response field '{field_name}' must be an array.")
    output: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"AI response {field_name}[{idx}] must be a string.")
        output.append(item)
    return output
