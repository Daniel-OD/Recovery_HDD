# AI report summarizer
from __future__ import annotations
import json, os
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class ReportAISummary:
    executive_summary:str

class ReportAI:
    def __init__(self,api_key=None,model="gpt-5"):
        self.client=OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model=model

    def summarize(self,data:dict):
        r=self.client.responses.create(model=self.model,input=json.dumps(data))
        return ReportAISummary(executive_summary=r.output_text)
