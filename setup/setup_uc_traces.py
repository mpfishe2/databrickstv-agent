"""Bind the MLflow experiment to Unity Catalog trace storage.

Usage:
    source .env  # or export DATABRICKS_CONFIG_PROFILE=your-profile
    python -m setup.setup_uc_traces
"""
import os
import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = os.getenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID")

catalog = os.environ.get("CATALOG")
if not catalog:
    raise RuntimeError("CATALOG is not set. Copy .env.example to .env and fill in the required values.")

schema = os.environ.get("SCHEMA", "databrickstv")
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/databrickstv-agent")
trace_schema = "mlflow_traces"

experiment = mlflow.set_experiment(
    experiment_name=experiment_name,
    trace_location=UnityCatalog(
        catalog_name=catalog,
        schema_name=trace_schema,
        table_prefix=schema,
    ),
)
print(f"Experiment bound to UC: {experiment.name} (ID: {experiment.experiment_id})")
print("Tables created:")
print(f"  - {catalog}.{trace_schema}.{schema}_otel_spans")
print(f"  - {catalog}.{trace_schema}.{schema}_otel_annotations")
print(f"  - {catalog}.{trace_schema}.{schema}_otel_logs")
print(f"  - {catalog}.{trace_schema}.{schema}_otel_metrics")
