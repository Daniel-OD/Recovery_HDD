# AI journal interpreter module
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
from typing import Any, Iterable

SYSTEM_PROMPT = "Journal interpretation assistant. Be conservative. Return JSON."

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {"type": "array"},
        "notes": {"type": "array"}
    },
    "required": ["events","notes"]
}

@dataclass
class JournalCandidate:
    inode_nr:int

@dataclass
class InterpretedJournalEvent:
    inode_nr:int
    candidate_name:str
    confidence:float

@dataclass
class JournalInterpretationResult:
    events:list[InterpretedJournalEvent]
    notes:list[str]

class JournalInterpreter:
    def __init__(self, api_key=None, model="gpt-5"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: openai. Install with `pip install openai`."
            ) from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def interpret(self, items:Iterable[JournalCandidate]):
        payload={"items":[asdict(x) for x in items]}
        r=self.client.responses.create(model=self.model,input=json.dumps(payload))
        data=json.loads(r.output_text)
        return JournalInterpretationResult(events=[],notes=data.get("notes",[]))
