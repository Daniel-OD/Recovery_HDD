# AI report summarizer
from __future__ import annotations
import json, os
from dataclasses import dataclass

@dataclass
class ReportAISummary:
    executive_summary:str

class ReportAI:
    def __init__(self,api_key=None,model="gpt-5"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: openai. Install with `pip install openai`."
            ) from exc
        self.client=OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model=model

    def summarize(self,data:dict):
        r=self.client.responses.create(model=self.model,input=json.dumps(data))
        return ReportAISummary(executive_summary=r.output_text)
