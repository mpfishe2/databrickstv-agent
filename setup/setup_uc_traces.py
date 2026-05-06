"""Bind the MLflow experiment to Unity Catalog trace storage.

Run once from local (not from the app):
    DATABRICKS_CONFIG_PROFILE=fevm-labelbricks-test python setup_uc_traces.py
"""
import os
import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = os.getenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID")

experiment = mlflow.set_experiment(
    experiment_name="/Shared/databrickstv-agent-v2",
    trace_location=UnityCatalog(
        catalog_name="labelbricks_test_catalog",
        schema_name="mlflow_traces",
        table_prefix="databrickstv",
    ),
)
print(f"Experiment bound to UC: {experiment.name} (ID: {experiment.experiment_id})")
print("Tables created:")
print("  - labelbricks_test_catalog.mlflow_traces.databrickstv_otel_spans")
print("  - labelbricks_test_catalog.mlflow_traces.databrickstv_otel_annotations")
print("  - labelbricks_test_catalog.mlflow_traces.databrickstv_otel_logs")
print("  - labelbricks_test_catalog.mlflow_traces.databrickstv_otel_metrics")
