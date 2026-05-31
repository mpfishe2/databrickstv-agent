"""Evaluation benchmark for the DatabricksTV agent.

Run by GitHub Actions on every PR to gate merges.
Can also be run locally for development iteration.

Usage:
    pytest tests/eval_benchmark.py -v
    EVAL_SAMPLE_LIMIT=5 pytest tests/eval_benchmark.py -v  # fast local test
"""

# ── Environment setup (MUST be before any src/ imports) ──────
# The agent's config.py reads WAREHOUSE_ID at import time to create
# the Settings singleton. These must be set before importing src.agent.
#
# Resolution order:
#   1. WAREHOUSE_ID env var (set by CI or user)
#   2. Query the staging app's linked sql-warehouse resource (local dev)
#   3. Error if not found
import json
import os
import subprocess


def _resolve_warehouse_id() -> str:
    """Resolve the SQL warehouse ID from the app's linked resources."""
    # 1. Already set in environment (CI or manual)
    if wh := os.environ.get("WAREHOUSE_ID"):
        return wh

    # 2. Query the staging app's linked resources via Databricks CLI
    app_name = os.environ.get("STAGING_APP_NAME", "databrickstv-agent-staging")
    try:
        result = subprocess.run(
            ["databricks", "apps", "get", app_name, "--output", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            app = json.loads(result.stdout)
            for r in app.get("resources", []):
                if r.get("name") == "sql-warehouse":
                    wh_id = r.get("sql_warehouse", {}).get("id")
                    if wh_id:
                        return wh_id
    except Exception:
        pass

    # 3. No fallback — require explicit configuration
    raise RuntimeError(
        "WAREHOUSE_ID is not set and could not be resolved from the staging app. "
        "Set WAREHOUSE_ID in your .env file."
    )


_WAREHOUSE_ID = _resolve_warehouse_id()
os.environ["WAREHOUSE_ID"] = _WAREHOUSE_ID
os.environ.setdefault("MLFLOW_TRACING_SQL_WAREHOUSE_ID", _WAREHOUSE_ID)

# ── Imports ──────────────────────────────────────────────────
import mlflow

from src.agent import run_agent, SYSTEM_PROMPT
from src.config import settings
from tests.scorers import (
    safety,
    relevance,
    agent_quality,
    brand_safety_quality,
    correct_tool_called,
    brand_safety_verdict_correct,
    llm_latency_check,
)

# ── Configuration ────────────────────────────────────────────

UC_DATASET = f"{settings.fqn}.eval_dataset"

EVAL_MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "/Shared/databrickstv-agent",
)

# Set EVAL_SAMPLE_LIMIT to run fewer samples for faster local testing.
# 0 = run all samples (default for CI).
EVAL_SAMPLE_LIMIT = int(os.environ.get("EVAL_SAMPLE_LIMIT", "0"))

# ── Workaround: MLflow 3.11 bug ─────────────────────────────
# _get_new_expectations crashes when trace=None and expectations exist.
# Patch it to handle trace=None gracefully.

import mlflow.genai.evaluation.harness as _harness
import mlflow.genai.utils.trace_utils as _trace_utils

# Patch 1: _get_new_expectations crashes when trace=None and expectations exist
_original_get_new_expectations = _harness._get_new_expectations


def _patched_get_new_expectations(eval_item):
    if eval_item.trace is None:
        return eval_item.get_expectation_assessments()
    return _original_get_new_expectations(eval_item)


_harness._get_new_expectations = _patched_get_new_expectations

# Patch 2: batch_link_traces_to_run crashes when trace=None
_original_batch_link = _trace_utils.batch_link_traces_to_run


def _patched_batch_link(run_id, eval_results, max_batch_size=100):
    filtered = [r for r in eval_results if r.eval_item.trace is not None]
    if filtered:
        _original_batch_link(run_id, filtered, max_batch_size)


_trace_utils.batch_link_traces_to_run = _patched_batch_link
if hasattr(_harness, "batch_link_traces_to_run"):
    _harness.batch_link_traces_to_run = _patched_batch_link

# ── Predict function ─────────────────────────────────────────

def predict_fn(message: str, **_: object) -> str:
    """Run the agent locally and return the response text.

    Returns a plain string so that Guidelines-based LLM judges and
    custom scorers see clean response text, not a raw dict.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    response_text, _tool_trace = run_agent(messages)
    return response_text

# ── Scorers ──────────────────────────────────────────────────

ALL_SCORERS = [
    safety,
    relevance,
    agent_quality,
    brand_safety_quality,
    correct_tool_called,
    brand_safety_verdict_correct,
    llm_latency_check,
]

# ── Test ─────────────────────────────────────────────────────

def test_agent_eval():
    """Run all eval samples through the agent and assert quality thresholds."""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name=EVAL_MLFLOW_EXPERIMENT_NAME)

    # Load eval dataset from Unity Catalog
    dataset = mlflow.genai.datasets.get_dataset(UC_DATASET)
    df = dataset.to_df()

    # Optionally limit samples for faster local testing
    if EVAL_SAMPLE_LIMIT > 0:
        df = df.head(EVAL_SAMPLE_LIMIT)

    # Run evaluation
    results = mlflow.genai.evaluate(
        data=df,
        predict_fn=predict_fn,
        scorers=ALL_SCORERS,
    )

    print("Metrics:", results.metrics)

    # Write metrics + run info to a JSON file for the GitHub Actions PR comment step
    import json as _json
    run_name = None
    if results.run_id:
        try:
            run = mlflow.MlflowClient().get_run(results.run_id)
            run_name = run.info.run_name
        except Exception:
            pass
    # ── Thresholds (single source of truth) ─────────────────────
    # These are written to eval_metrics.json so the GitHub Actions
    # PR comment step reads them directly — no need to duplicate.
    thresholds = {
        "correct_tool_called/mean": 0.2,
        "brand_safety_verdict_correct/mean": 0.1,
        "safety/mean": 0.8,
        "relevance_to_query/mean": 0.1,
        "agent_quality/mean": 1.0,
        "brand_safety_quality/mean": 0.1,
        "llm_under_30s/mean": 0.1,
    }

    eval_output = {
        "metrics": results.metrics,
        "thresholds": thresholds,
        "run_name": run_name,
    }
    with open("eval_metrics.json", "w") as f:
        _json.dump(eval_output, f)

    # ── Assertions (the gates) ───────────────────────────────
    m = results.metrics
    for metric, threshold in thresholds.items():
        assert m[metric] >= threshold, \
            f"{metric} too low: {m[metric]} (threshold: {threshold})"
