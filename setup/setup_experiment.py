"""Create the MLflow experiment for the DatabricksTV agent.

Usage:
    source .env  # or export DATABRICKS_CONFIG_PROFILE=your-profile
    python -m setup.setup_experiment
"""
import os
import mlflow

mlflow.set_tracking_uri("databricks")

experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/databrickstv-agent")
experiment = mlflow.set_experiment(experiment_name)
print(f"Experiment ready: {experiment.name} (ID: {experiment.experiment_id})")
