"""Offline evaluation — run predictions and score results.

Usage (local):
    source .env  # or export DATABRICKS_CONFIG_PROFILE=your-profile
    python -m tests.eval_runner

This runner works locally by:
1. Running all 30 predictions through the agent
2. Performing brand safety verdict checks (deterministic)
3. Logging results as an MLflow run with metrics

For full LLM-judge scoring (Safety, Relevance, Guidelines), run this as a
Databricks notebook where mlflow.genai.evaluate() has native trace support.
"""
import json
import os
import time
import mlflow

from src.agent import run_agent, SYSTEM_PROMPT
from tests.eval_data import eval_data

mlflow.set_tracking_uri("databricks")
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/databrickstv-agent")
mlflow.set_experiment(experiment_name)


def predict_fn(message: str) -> dict:
    """Run the agent on a single message and return the response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    response_text, tool_trace = run_agent(messages)
    return {"response": response_text, "tool_trace": tool_trace}


if __name__ == "__main__":
    start = time.time()

    # Step 1: Run all predictions
    print(f"Running {len(eval_data)} predictions...")
    results = []
    for i, sample in enumerate(eval_data):
        msg = sample["inputs"]["message"]
        print(f"  [{i+1}/{len(eval_data)}] {msg[:60]}...")
        try:
            output = predict_fn(msg)
            results.append({"inputs": sample["inputs"], "expectations": sample.get("expectations", {}), "output": output})
            tools_called = [t["tool"] for t in output.get("tool_trace", [])]
            print(f"    -> {len(output['response'])} chars, tools: {tools_called}")
        except Exception as e:
            results.append({"inputs": sample["inputs"], "expectations": sample.get("expectations", {}), "output": {"response": f"ERROR: {e}", "tool_trace": []}})
            print(f"    ERROR: {e}")

    elapsed_predict = time.time() - start
    print(f"\nAll {len(results)} predictions complete in {elapsed_predict:.0f}s")

    # Step 2: Score — Tool call correctness (deterministic)
    print("\n=== Tool Call Correctness ===")
    tool_correct = 0
    tool_total = 0
    for r in results:
        expected_tools = r["expectations"].get("expected_tools", [])
        if not expected_tools:
            continue
        tool_total += 1
        called = [t["tool"] for t in r["output"].get("tool_trace", [])]
        all_found = all(t in called for t in expected_tools)
        if all_found:
            tool_correct += 1
            status = "PASS"
        else:
            status = "FAIL"
        msg = r["inputs"]["message"][:50]
        print(f"  [{status}] {msg}...  expected={expected_tools} called={called}")

    tool_accuracy = tool_correct / tool_total if tool_total else 0
    print(f"\nTool call accuracy: {tool_correct}/{tool_total} ({tool_accuracy:.0%})")

    # Step 3: Score — Brand safety verdict (deterministic)
    print("\n=== Brand Safety Verdict ===")
    bs_correct = 0
    bs_total = 0
    for r in results:
        expected_verdict = r["expectations"].get("expected_verdict", "")
        if not expected_verdict:
            continue
        bs_total += 1
        resp_upper = r["output"]["response"].upper()
        if "UNSAFE" in resp_upper:
            actual = "unsafe"
        elif "SAFE" in resp_upper:
            actual = "safe"
        else:
            actual = "unknown"
        match = actual == expected_verdict
        if match:
            bs_correct += 1
        status = "PASS" if match else "FAIL"
        msg = r["inputs"]["message"][:50]
        print(f"  [{status}] {msg}...  expected={expected_verdict} actual={actual}")

    bs_accuracy = bs_correct / bs_total if bs_total else 0
    print(f"\nBrand safety verdict accuracy: {bs_correct}/{bs_total} ({bs_accuracy:.0%})")

    # Step 4: Summary metrics
    errors = sum(1 for r in results if r["output"]["response"].startswith("ERROR"))
    total_time = time.time() - start

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples:              {len(results)}")
    print(f"Successful predictions:     {len(results) - errors}")
    print(f"Errors:                     {errors}")
    print(f"Tool call accuracy:         {tool_correct}/{tool_total} ({tool_accuracy:.0%})")
    print(f"Brand safety verdict acc:   {bs_correct}/{bs_total} ({bs_accuracy:.0%})")
    print(f"Total time:                 {total_time:.0f}s")
    print(f"Avg prediction time:        {elapsed_predict/len(results):.1f}s")

    # Step 5: Log as MLflow run
    with mlflow.start_run(run_name="eval_run_local"):
        mlflow.log_metric("num_samples", len(results))
        mlflow.log_metric("num_errors", errors)
        mlflow.log_metric("tool_call_accuracy", tool_accuracy)
        mlflow.log_metric("brand_safety_verdict_accuracy", bs_accuracy)
        mlflow.log_metric("avg_prediction_time_s", elapsed_predict / len(results))
        mlflow.log_metric("total_time_s", total_time)

        # Log full results as artifact
        with open("/tmp/eval_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact("/tmp/eval_results.json")

    print(f"\nResults logged to MLflow experiment {experiment_name}")
