# Databricks notebook source

# MAGIC %md
# MAGIC # Sync Failing Production Traces to Eval Dataset
# MAGIC
# MAGIC Closes the feedback loop between production monitoring and the eval pipeline.
# MAGIC Queries production traces where a monitor flagged a failure, then merges them
# MAGIC into the UC eval dataset so they become regression tests in the next PR eval.
# MAGIC
# MAGIC **Flow:**
# MAGIC ```
# MAGIC Production app → Monitors score traces → This notebook queries failures
# MAGIC → Merges into eval dataset → Next PR eval includes these as test cases
# MAGIC ```
# MAGIC
# MAGIC **Parameters (Databricks widgets):**
# MAGIC - `catalog` — Unity Catalog name (e.g., `labelbricks_test_catalog`)
# MAGIC - `schema` — Schema name (default: `databrickstv`)
# MAGIC - `experiment_name` — MLflow experiment path (default: `/Shared/databrickstv-agent`)
# MAGIC - `max_results` — Max traces to query per monitor (default: `200`)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog Name")
dbutils.widgets.text("schema", "databrickstv", "Schema Name")
dbutils.widgets.text("experiment_name", "/Shared/databrickstv-agent", "MLflow Experiment")
dbutils.widgets.text("max_results", "200", "Max Results Per Monitor")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")
max_results = int(dbutils.widgets.get("max_results"))

if not catalog:
    raise ValueError("The 'catalog' widget must be set.")

UC_DATASET = f"{catalog}.{schema}.eval_dataset"
print(f"Catalog:    {catalog}")
print(f"Schema:     {schema}")
print(f"Dataset:    {UC_DATASET}")
print(f"Experiment: {experiment_name}")
print(f"Max results per monitor: {max_results}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ensure the eval dataset exists
# MAGIC
# MAGIC If the dataset doesn't exist yet, create it and populate it with the 30
# MAGIC baseline samples. This makes the notebook self-contained — you don't need
# MAGIC to run `create_eval_dataset` first.

# COMMAND ----------

import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)

try:
    dataset = mlflow.genai.datasets.get_dataset(UC_DATASET)
    df = dataset.to_df()
    print(f"Eval dataset exists with {len(df)} records")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "TABLE_OR_VIEW_NOT_FOUND" in str(e) or "does not exist" in str(e).lower():
        print(f"Dataset not found. Creating and populating with baseline samples...")
        dbutils.notebook.run("create_eval_dataset", timeout_seconds=300, arguments={
            "catalog": catalog,
            "schema": schema,
            "experiment_name": experiment_name,
        })
        dataset = mlflow.genai.datasets.get_dataset(UC_DATASET)
        df = dataset.to_df()
        print(f"Created eval dataset with {len(df)} records")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Query failing production traces via SQL
# MAGIC
# MAGIC The three production monitors write assessment annotations to the UC trace tables:
# MAGIC - `prod_safety` — 100% sample rate
# MAGIC - `prod_agent_quality` — 25% sample rate
# MAGIC - `prod_brand_safety_quality` — 50% sample rate
# MAGIC
# MAGIC We query the `_otel_annotations` table for assessments with `value = 'no'`
# MAGIC (failure), then join against `_otel_spans` to extract the user's input message.
# MAGIC
# MAGIC **Why SQL instead of `mlflow.search_traces()`?**
# MAGIC The `assessments.*` filter syntax is not supported by the Databricks MLflow
# MAGIC REST API. Querying the UC tables directly is reliable and fast.

# COMMAND ----------

import json

trace_schema = "mlflow_traces"
table_prefix = schema

annotations_table = f"{catalog}.{trace_schema}.{table_prefix}_otel_annotations"
spans_table = f"{catalog}.{trace_schema}.{table_prefix}_otel_spans"

MONITOR_NAMES = [
    "prod_safety",
    "prod_agent_quality",
    "prod_brand_safety_quality",
]

# Show table schemas for debugging
print(f"--- {annotations_table} columns ---")
for col in spark.sql(f"DESCRIBE {annotations_table}").collect():
    print(f"  {col.col_name}: {col.data_type}")

print(f"\n--- {spans_table} columns ---")
for col in spark.sql(f"DESCRIBE {spans_table}").collect():
    print(f"  {col.col_name}: {col.data_type}")

# Find trace IDs where any monitor flagged a failure
monitor_list = ", ".join(f"'{m}'" for m in MONITOR_NAMES)

failing_traces_query = f"""
SELECT DISTINCT a.target_id AS trace_id, a.name AS monitor_name
FROM {annotations_table} a
WHERE a.name IN ({monitor_list})
  AND a.value::STRING = 'no'
ORDER BY trace_id
LIMIT {max_results}
"""

failing_df = spark.sql(failing_traces_query)
failing_rows = failing_df.collect()

# Count by monitor
counts = {}
trace_ids = set()
for row in failing_rows:
    counts[row.monitor_name] = counts.get(row.monitor_name, 0) + 1
    trace_ids.add(row.trace_id)

print(f"Found {len(trace_ids)} unique failing traces:")
for name in MONITOR_NAMES:
    print(f"  {name}: {counts.get(name, 0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Extract user messages from failing traces
# MAGIC
# MAGIC The eval dataset expects records with `{"inputs": {"message": "..."}}`.
# MAGIC We query the root span of each failing trace to extract the user's input,
# MAGIC then deduplicate by message text.

# COMMAND ----------

records = []

if trace_ids:
    # Get root spans for failing traces — these contain the request input
    trace_id_list = ", ".join(f"'{t}'" for t in trace_ids)
    spans_query = f"""
    SELECT trace_id, attributes:`mlflow.spanInputs` AS span_inputs
    FROM {spans_table}
    WHERE trace_id IN ({trace_id_list})
      AND parent_span_id IS NULL
    """
    spans_df = spark.sql(spans_query)

    seen_messages = set()
    for row in spans_df.collect():
        try:
            inputs_raw = row.span_inputs
            if not inputs_raw:
                continue
            inputs = json.loads(str(inputs_raw)) if isinstance(inputs_raw, str) else inputs_raw

            # Extract user message from the messages list
            messages = inputs.get("messages", []) if isinstance(inputs, dict) else []
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    if user_msg and user_msg not in seen_messages:
                        seen_messages.add(user_msg)
                        records.append({"inputs": {"message": user_msg}})
                    break
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

print(f"{len(records)} unique records to merge (from {len(trace_ids)} failing traces)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Merge failing traces into the eval dataset
# MAGIC
# MAGIC `merge_records` handles deduplication — records already in the dataset
# MAGIC won't be added again.

# COMMAND ----------

if not records:
    print("No failing traces to merge. Dataset unchanged.")
else:
    dataset.merge_records(records)
    df = dataset.to_df()
    print(f"Merged {len(records)} records. Dataset now has {len(df)} total records.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Any merged traces will be included as regression tests the next time
# MAGIC `tests/eval_benchmark.py` runs (on the next PR).
