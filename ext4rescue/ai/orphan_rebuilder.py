from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

SYSTEM_PROMPT = """\
You are a digital forensics assistant helping reorganize orphaned recovered files.
Your job:
- Suggest likely original directory structure for orphaned files recovered from an ext4 filesystem.
- Be conservative.
- Do not invent unrealistic names.
- Prefer grouping by timestamps, file types, size patterns, and known media naming conventions.
- Preserve original file extensions.
- Return structured JSON only.
Rules:
- If confidence is low, keep the original fallback filename and place it in a generic orphan bucket.
- Never claim certainty when only a weak pattern exists.
- Keep names filesystem-safe.
- Do not use slashes inside file names.
- Suggest folders like photos/YYYY-MM-DD, videos/YYYY-MM-DD, documents/, archives/, audio/, raw/, misc/ when appropriate.
"""

DEFAULT_MAX_ITEMS = 300

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "folder_suggestion": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                    "inode_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "group_id",
                    "folder_suggestion",
                    "reason",
                    "confidence",
                    "inode_numbers",
                ],
                "additionalProperties": False,
            },
        },
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "inode_nr": {"type": "integer"},
                    "suggested_parent": {"type": "string"},
                    "suggested_name": {"type": "string"},
                    "suggested_path": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                    "keep_original_name": {"type": "boolean"},
                },
                "required": [
                    "inode_nr",
                    "suggested_parent",
                    "suggested_name",
                    "suggested_path",
                    "confidence",
                    "reason",
                    "keep_original_name",
                ],
                "additionalProperties": False,
            },
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["groups", "files", "notes"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class OrphanRecord:
    inode_nr: int
    fallback_name: str
    extension: str | None = None
    size: int | None = None
    mtime: int | None = None
    ctime: int | None = None
    crtime: int | None = None
    source: str = "orphan"
    mime_hint: str | None = None
    magic_hint: str | None = None
    md5: str | None = None
    physical_regions: list[tuple[int, int]] | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "inode_nr": self.inode_nr,
            "fallback_name": self.fallback_name,
            "extension": self.extension,
            "size": self.size,
            "mtime": self.mtime,
            "ctime": self.ctime,
            "crtime": self.crtime,
            "source": self.source,
            "mime_hint": self.mime_hint,
            "magic_hint": self.magic_hint,
            "md5": self.md5,
            "physical_regions": self.physical_regions or [],
        }


@dataclass(slots=True)
class AISuggestedGroup:
    group_id: str
    folder_suggestion: str
    reason: str
    confidence: float
    inode_numbers: list[int]


@dataclass(slots=True)
class AISuggestedFile:
    inode_nr: int
    suggested_parent: str
    suggested_name: str
    suggested_path: str
    confidence: float
    reason: str
    keep_original_name: bool


@dataclass(slots=True)
class OrphanRebuildResult:
    groups: list[AISuggestedGroup]
    files: list[AISuggestedFile]
    notes: list[str]
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [asdict(g) for g in self.groups],
            "files": [asdict(f) for f in self.files],
            "notes": list(self.notes),
        }


class OrphanRebuilder:
    """AI-assisted orphan file organizer."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5",
        confidence_threshold: float = 0.80,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: openai. Install with `pip install openai`."
            ) from exc
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.confidence_threshold = confidence_threshold

    def suggest_structure(
        self,
        orphans: Iterable[OrphanRecord],
        *,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> OrphanRebuildResult:
        orphan_list = list(orphans)
        if not orphan_list:
            return OrphanRebuildResult(
                groups=[],
                files=[],
                notes=["No orphan files provided."],
            )
        if len(orphan_list) > max_items:
            raise ValueError(
                f"Too many orphan items: {len(orphan_list)} (max_items={max_items})."
            )

        payload = {
            "task": "suggest_likely_structure_for_orphans",
            "orphans": [x.to_prompt_dict() for x in orphan_list],
            "instructions": {
                "preserve_extensions": True,
                "be_conservative": True,
                "generic_low_confidence_bucket": "_orphans/low_confidence",
                "suggest_date_based_folders_when_supported": True,
                "max_confidence": 1.0,
                "min_confidence": 0.0,
            },
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "orphan_rebuild_result",
                    "strict": True,
                    "schema": JSON_SCHEMA,
                },
            },
        )

        raw_text = response.output_text
        parsed = _parse_result_json(raw_text)
        groups = _parse_groups(parsed["groups"])
        files = _parse_files(parsed["files"])
        notes = _parse_notes(parsed["notes"])
        return OrphanRebuildResult(
            groups=groups,
            files=files,
            notes=notes,
            raw_response=raw_text,
        )

    def accepted_suggestions(
        self,
        result: OrphanRebuildResult,
    ) -> list[AISuggestedFile]:
        return [f for f in result.files if f.confidence >= self.confidence_threshold]

    def low_confidence_suggestions(
        self,
        result: OrphanRebuildResult,
    ) -> list[AISuggestedFile]:
        return [f for f in result.files if f.confidence < self.confidence_threshold]


def _parse_result_json(raw_text: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI response is not valid JSON: {exc.msg}.") from exc
    if not isinstance(data, dict):
        raise ValueError("AI response must be a JSON object.")
    if set(data.keys()) != {"groups", "files", "notes"}:
        raise ValueError("AI response must contain exactly: groups, files, notes.")
    return data


def _parse_groups(raw_groups: Any) -> list[AISuggestedGroup]:
    if not isinstance(raw_groups, list):
        raise ValueError("AI response field 'groups' must be an array.")
    groups: list[AISuggestedGroup] = []
    for idx, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            raise ValueError(f"AI response groups[{idx}] must be an object.")
        required = {"group_id", "folder_suggestion", "reason", "confidence", "inode_numbers"}
        extra = set(group.keys()) - required
        missing = required - set(group.keys())
        if missing:
            raise ValueError(
                f"AI response groups[{idx}] missing field(s): {', '.join(sorted(missing))}."
            )
        if extra:
            raise ValueError(
                f"AI response groups[{idx}] has unexpected field(s): {', '.join(sorted(extra))}."
            )
        inode_numbers = group["inode_numbers"]
        if not isinstance(group["group_id"], str):
            raise ValueError(f"AI response groups[{idx}].group_id must be a string.")
        if not isinstance(group["folder_suggestion"], str):
            raise ValueError(f"AI response groups[{idx}].folder_suggestion must be a string.")
        if not isinstance(group["reason"], str):
            raise ValueError(f"AI response groups[{idx}].reason must be a string.")
        if not isinstance(group["confidence"], (int, float)):
            raise ValueError(f"AI response groups[{idx}].confidence must be a number.")
        if not isinstance(inode_numbers, list) or any(not isinstance(x, int) for x in inode_numbers):
            raise ValueError(f"AI response groups[{idx}].inode_numbers must be an array of integers.")
        groups.append(
            AISuggestedGroup(
                group_id=group["group_id"],
                folder_suggestion=group["folder_suggestion"],
                reason=group["reason"],
                confidence=float(group["confidence"]),
                inode_numbers=inode_numbers,
            )
        )
    return groups


def _parse_files(raw_files: Any) -> list[AISuggestedFile]:
    if not isinstance(raw_files, list):
        raise ValueError("AI response field 'files' must be an array.")
    files: list[AISuggestedFile] = []
    for idx, entry in enumerate(raw_files):
        if not isinstance(entry, dict):
            raise ValueError(f"AI response files[{idx}] must be an object.")
        required = {
            "inode_nr", "suggested_parent", "suggested_name", "suggested_path",
            "confidence", "reason", "keep_original_name",
        }
        extra = set(entry.keys()) - required
        missing = required - set(entry.keys())
        if missing:
            raise ValueError(
                f"AI response files[{idx}] missing field(s): {', '.join(sorted(missing))}."
            )
        if extra:
            raise ValueError(
                f"AI response files[{idx}] has unexpected field(s): {', '.join(sorted(extra))}."
            )
        if not isinstance(entry["inode_nr"], int):
            raise ValueError(f"AI response files[{idx}].inode_nr must be an integer.")
        if not isinstance(entry["suggested_parent"], str):
            raise ValueError(f"AI response files[{idx}].suggested_parent must be a string.")
        if not isinstance(entry["suggested_name"], str):
            raise ValueError(f"AI response files[{idx}].suggested_name must be a string.")
        if not isinstance(entry["suggested_path"], str):
            raise ValueError(f"AI response files[{idx}].suggested_path must be a string.")
        if not isinstance(entry["confidence"], (int, float)):
            raise ValueError(f"AI response files[{idx}].confidence must be a number.")
        if not isinstance(entry["reason"], str):
            raise ValueError(f"AI response files[{idx}].reason must be a string.")
        if not isinstance(entry["keep_original_name"], bool):
            raise ValueError(f"AI response files[{idx}].keep_original_name must be a boolean.")
        files.append(
            AISuggestedFile(
                inode_nr=entry["inode_nr"],
                suggested_parent=entry["suggested_parent"],
                suggested_name=entry["suggested_name"],
                suggested_path=entry["suggested_path"],
                confidence=float(entry["confidence"]),
                reason=entry["reason"],
                keep_original_name=entry["keep_original_name"],
            )
        )
    return files


def _parse_notes(raw_notes: Any) -> list[str]:
    if not isinstance(raw_notes, list):
        raise ValueError("AI response field 'notes' must be an array.")
    notes: list[str] = []
    for idx, note in enumerate(raw_notes):
        if not isinstance(note, str):
            raise ValueError(f"AI response notes[{idx}] must be a string.")
        notes.append(note)
    return notes
