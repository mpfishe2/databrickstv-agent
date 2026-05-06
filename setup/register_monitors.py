"""Register production monitors for the DatabricksTV agent.

Usage:
    python register_monitors.py

This registers scorers as production monitors that automatically
evaluate a sample of live traces in the MLflow experiment.
"""
import mlflow
from mlflow.genai.scorers import ScorerSamplingConfig

from tests.scorers import safety, agent_quality, brand_safety_quality

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/databrickstv-agent")

# Safety: evaluate 100% of traces
safety.register(name="prod_safety").start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0)
)
print("Registered prod_safety monitor (100% sample rate)")

# Agent quality: evaluate 25% of traces
agent_quality.register(name="prod_agent_quality").start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.25)
)
print("Registered prod_agent_quality monitor (25% sample rate)")

# Brand safety quality: evaluate 50% of traces
brand_safety_quality.register(name="prod_brand_safety_quality").start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5)
)
print("Registered prod_brand_safety_quality monitor (50% sample rate)")

print("\nAll production monitors registered successfully.")
