"""Create the MLflow experiment for the DatabricksTV agent."""
import mlflow

mlflow.set_tracking_uri("databricks")
experiment = mlflow.set_experiment("/Shared/databrickstv-agent")
print(f"Experiment ready: {experiment.name} (ID: {experiment.experiment_id})")
