import unittest
import os
from unittest.mock import mock_open, patch

from app.eval_trace import emit
from evals.runner import evaluate, mock_traces


class EvalRunnerTests(unittest.TestCase):
    def test_retrieval_and_memory_metrics_use_only_applicable_cases(self):
        cases = [
            {"id": "knowledge-a", "expected_tools": [{"name": "search", "arguments": {}}], "required_evidence": ["knowledge:x"]},
            {"id": "memory-b", "expected_tools": [{"name": "memory", "arguments": {}}], "required_evidence": ["memory:x"]},
        ]
        report = evaluate(cases, mock_traces(cases))
        self.assertEqual(report["metrics"]["rag_hit_at_3"], 1.0)
        self.assertEqual(report["metrics"]["memory_recall_rate"], 1.0)

    def test_controlled_trace_export_is_jsonl(self):
        old_path = os.environ.get("AI_EVAL_TRACE_FILE")
        os.environ["AI_EVAL_TRACE_FILE"] = "ignored.jsonl"
        stream = mock_open()
        try:
            with patch("app.eval_trace.Path.open", stream):
                emit("case-1", {"tool_calls": []}, "answer")
        finally:
            if old_path is None:
                os.environ.pop("AI_EVAL_TRACE_FILE", None)
            else:
                os.environ["AI_EVAL_TRACE_FILE"] = old_path
        self.assertIn('"execution_kind": "real_integration"', stream().write.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
