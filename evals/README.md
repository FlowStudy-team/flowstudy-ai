# FlowStudy Agent Eval

`python -m evals.runner --mock` verifies evaluator mechanics only. Its report
is explicitly `offline_mock` and is not a model result.

Index a frozen corpus before a real run:

```powershell
python -m evals.indexer --corpus evals/corpora/integration-snapshot-v1.json
```

For a real run, set `AI_EVAL_TRACE_FILE=artifacts/traces.jsonl` only in the
controlled integration target and send each Golden request with
`X-Eval-Case-Id`. Then run `python -m evals.runner --trace traces.jsonl --output
artifacts/current`. Each trace must contain `case_id`, `tool_calls`,
`final_answer`, `model_rounds`, `latency_ms`, provider token fields when the
provider returned them, and `execution_kind: real_integration`.

Compare only equivalent execution kinds:

```powershell
python -m evals.runner --trace current.json --baseline artifacts/baseline/run.json --output artifacts/current
```

The Golden Set is a manually reviewed fixture. Replace its manually marked
memory item with an anonymized controlled-environment snapshot before a release
run. `benchmarks.json` records public-benchmark scope and prevents unsuitable
tau2/SWE-bench results from being presented as FlowStudy results.
