"""Evaluate exported agent traces without an LLM judge.

Real traces are produced only by a controlled integration target.  The optional
``--mock`` mode exists to test this evaluator and is conspicuously labelled in
all artifacts; it must never be compared with real-model runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def _normal(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normal(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normal(item) for item in value]
    return value


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(math.ceil(len(ordered) * .95) - 1, len(ordered) - 1)]


def mock_traces(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic self-test data, deliberately not a model result."""
    return [{"case_id": case["id"], "final_answer": "[mock] " + " ".join(case["required_evidence"]), "tool_calls": [{**tool, "success": True, "latency_ms": 1.0} for tool in case["expected_tools"]], "model_rounds": 2, "latency_ms": 5.0, "provider_input_tokens": None, "provider_output_tokens": None, "execution_kind": "offline_mock"} for case in cases]


def evaluate(cases: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {trace["case_id"]: trace for trace in traces}
    rows: list[dict[str, Any]] = []
    for case in cases:
        trace = by_id.get(case["id"], {})
        actual = trace.get("tool_calls", [])
        expected = case["expected_tools"]
        selected = [call.get("name") for call in actual] == [call["name"] for call in expected]
        arguments = selected and all(_normal(call.get("arguments", {})) == _normal(want["arguments"]) for call, want in zip(actual, expected))
        executed = bool(actual) and all(call.get("success") is True for call in actual)
        answer = str(trace.get("final_answer", ""))
        evidence = all(item in answer or item in trace.get("evidence_ids", []) for item in case["required_evidence"])
        success = bool(selected and arguments and executed and evidence)
        rows.append({"case_id": case["id"], "task_success": success, "tool_selected": selected, "arguments_correct": bool(arguments), "tool_execution_success": executed, "rag_hit_at_3": evidence if "knowledge" in case["id"] else None, "memory_recall": evidence if "memory" in case["id"] else None, "mrr": 1.0 if evidence and "knowledge" in case["id"] else None, "trace": trace})
    def ratio(key: str) -> float:
        return round(sum(bool(row[key]) for row in rows) / len(rows), 4) if rows else 0.0
    def subset_ratio(key: str) -> float | None:
        subset = [row[key] for row in rows if row[key] is not None]
        return round(sum(bool(value) for value in subset) / len(subset), 4) if subset else None
    latency = [float(row["trace"].get("latency_ms", 0)) for row in rows if row["trace"]]
    tokens = [int(row["trace"].get("provider_input_tokens") or 0) + int(row["trace"].get("provider_output_tokens") or 0) for row in rows if row["task_success"] and row["trace"].get("provider_input_tokens") is not None]
    return {"cases": rows, "metrics": {"task_success_rate": ratio("task_success"), "tool_selection_accuracy": ratio("tool_selected"), "argument_accuracy": ratio("arguments_correct"), "tool_execution_success_rate": ratio("tool_execution_success"), "rag_hit_at_3": subset_ratio("rag_hit_at_3"), "rag_mrr": round(statistics.mean(row["mrr"] for row in rows if row["mrr"] is not None), 4) if any(row["mrr"] is not None for row in rows) else None, "memory_recall_rate": subset_ratio("memory_recall"), "avg_agent_turns": round(statistics.mean(row["trace"].get("model_rounds", 0) for row in rows), 3) if rows else 0.0, "avg_tool_calls": round(statistics.mean(len(row["trace"].get("tool_calls", [])) for row in rows), 3) if rows else 0.0, "p95_latency_ms": _p95(latency), "token_per_successful_task": round(statistics.mean(tokens), 2) if tokens else None}}


def markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = ["# FlowStudy Agent Eval", "", f"- Execution kind: `{report['execution_kind']}`", f"- Dataset: `{report['dataset_id']}`", f"- Generated: `{report['generated_at']}`", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value if value is not None else 'unavailable'} |" for key, value in metrics.items())
    lines.extend(["", "## Cases", "", "| Case | Success | Tool | Args | Evidence |", "|---|---:|---:|---:|---:|"])
    lines.extend(f"| {row['case_id']} | {row['task_success']} | {row['tool_selected']} | {row['arguments_correct']} | {row['rag_hit_at_3'] if row['rag_hit_at_3'] is not None else row['memory_recall']} |" for row in report["cases"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=ROOT / "golden" / "flowstudy-golden-v1.json")
    parser.add_argument("--trace", type=Path, help="JSON list of real integration traces")
    parser.add_argument("--mock", action="store_true", help="run deterministic evaluator self-test only")
    parser.add_argument("--baseline", type=Path, help="prior run.json; only same execution kind may be compared")
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval"))
    args = parser.parse_args()
    if bool(args.trace) == bool(args.mock):
        parser.error("choose exactly one of --trace or --mock")
    dataset = _load(args.golden)
    traces = mock_traces(dataset["cases"]) if args.mock else _load(args.trace)
    kinds = {trace.get("execution_kind", "real_integration") for trace in traces}
    report = evaluate(dataset["cases"], traces)
    report.update({"dataset_id": dataset["dataset_id"], "execution_kind": ",".join(sorted(kinds)), "generated_at": datetime.now(timezone.utc).isoformat()})
    if args.baseline:
        baseline = _load(args.baseline)
        if baseline.get("execution_kind") != report["execution_kind"]:
            parser.error("baseline execution kind differs; mock and real results cannot be compared")
        quality_keys = ("task_success_rate", "tool_selection_accuracy", "argument_accuracy", "tool_execution_success_rate", "rag_hit_at_3", "rag_mrr", "memory_recall_rate")
        deltas = {key: round(report["metrics"][key] - baseline["metrics"][key], 4) for key in quality_keys if report["metrics"].get(key) is not None and baseline["metrics"].get(key) is not None}
        report["comparison"] = {"baseline": str(args.baseline), "deltas": deltas, "quality_gate_passed": all(value >= -0.03 for value in deltas.values())}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    if report.get("comparison", {}).get("quality_gate_passed") is False:
        return 2
    print(args.output / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
