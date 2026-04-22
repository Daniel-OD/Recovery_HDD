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

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [asdict(g) for g in self.groups],
            "files": [asdict(f) for f in self.files],
            "notes": list(self.notes),
        }


class OrphanRebuilder:
    """
    AI-assisted orphan file organizer.
    This class does not modify any files. It only produces suggestions.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5",
        confidence_threshold: float = 0.80,
    ) -> None:
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
        max_items: int = 200,
    ) -> OrphanRebuildResult:
        orphan_list = list(orphans)
        if not orphan_list:
            return OrphanRebuildResult(
                groups=[],
                files=[],
                notes=["No orphan files provided."],
            )

        orphan_list = orphan_list[:max_items]
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

        raw = json.loads(response.output_text)
        groups = [AISuggestedGroup(**g) for g in raw["groups"]]
        files = [AISuggestedFile(**f) for f in raw["files"]]
        notes = list(raw["notes"])
        return OrphanRebuildResult(groups=groups, files=files, notes=notes)

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
