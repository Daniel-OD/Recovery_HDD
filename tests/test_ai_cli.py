import argparse
import json
from dataclasses import dataclass

import pytest

from ext4rescue import cli
from ext4rescue.ai.journal_interpreter import InterpretedJournalEvent, JournalInterpretationResult


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ai_journal_rejects_more_than_max_items(tmp_path: pytest.TempPathFactory) -> None:
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    payload = {"items": [{"inode_nr": i} for i in range(301)]}
    _write_json(input_path, payload)

    args = argparse.Namespace(
        input_json=str(input_path),
        output_json=str(output_path),
        model="gpt-5",
        debug=False,
    )
    with pytest.raises(SystemExit):
        cli._cmd_ai_journal(args)


def test_ai_journal_dataclass_mapping_error_includes_index(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    payload = {"items": [{}]}
    _write_json(input_path, payload)

    args = argparse.Namespace(
        input_json=str(input_path),
        output_json=str(output_path),
        model="gpt-5",
        debug=False,
    )
    with pytest.raises(SystemExit):
        cli._cmd_ai_journal(args)
    stderr = capsys.readouterr().err
    assert "journal item[0] invalid mapping" in stderr
    assert "missing required field(s): inode_nr" in stderr


def test_ai_journal_debug_prints_raw_response_and_writes_payload(tmp_path, monkeypatch, capsys) -> None:
    import ext4rescue.ai as ai_module

    class FakeJournalInterpreter:
        def __init__(self, model: str) -> None:
            assert model == "gpt-5"

        def interpret(self, items, *, max_items: int):
            assert max_items == 300
            assert len(list(items)) == 1
            return JournalInterpretationResult(
                events=[
                    InterpretedJournalEvent(
                        inode_nr=10,
                        candidate_name="photo.jpg",
                        confidence=0.91,
                    )
                ],
                notes=["ok"],
                raw_response='{"events":[{"inode_nr":10,"candidate_name":"photo.jpg","confidence":0.91}],"notes":["ok"]}',
            )

    monkeypatch.setattr(ai_module, "JournalInterpreter", FakeJournalInterpreter)

    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    _write_json(input_path, {"items": [{"inode_nr": 10}]})
    args = argparse.Namespace(
        input_json=str(input_path),
        output_json=str(output_path),
        model="gpt-5",
        debug=True,
    )

    cli._cmd_ai_journal(args)
    stdout = capsys.readouterr().out
    assert "Raw AI response:" in stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["notes"] == ["ok"]
    assert payload["events"][0]["candidate_name"] == "photo.jpg"


def test_ai_report_rejects_oversized_top_level_list(tmp_path) -> None:
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    _write_json(input_path, {"items": list(range(301))})

    args = argparse.Namespace(
        input_json=str(input_path),
        output_json=str(output_path),
        model="gpt-5",
        debug=False,
    )
    with pytest.raises(SystemExit):
        cli._cmd_ai_report(args)


def test_ai_report_writes_extended_fields_and_debug_output(tmp_path, monkeypatch, capsys) -> None:
    import ext4rescue.ai as ai_module

    @dataclass
    class FakeSummary:
        executive_summary: str
        highlights: list[str]
        warnings: list[str]
        next_steps: list[str]
        operator_notes: list[str]
        raw_response: str

    class FakeReportAI:
        def __init__(self, model: str) -> None:
            assert model == "gpt-5"

        def summarize(self, data, *, max_items: int):
            assert isinstance(data, dict)
            assert max_items == 300
            return FakeSummary(
                executive_summary="summary",
                highlights=["h1"],
                warnings=["w1"],
                next_steps=["n1"],
                operator_notes=["o1"],
                raw_response='{"executive_summary":"summary"}',
            )

    monkeypatch.setattr(ai_module, "ReportAI", FakeReportAI)

    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    _write_json(input_path, {"report": "data"})
    args = argparse.Namespace(
        input_json=str(input_path),
        output_json=str(output_path),
        model="gpt-5",
        debug=True,
    )

    cli._cmd_ai_report(args)
    stdout = capsys.readouterr().out
    assert "Raw AI response:" in stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "executive_summary": "summary",
        "highlights": ["h1"],
        "warnings": ["w1"],
        "next_steps": ["n1"],
        "operator_notes": ["o1"],
    }
