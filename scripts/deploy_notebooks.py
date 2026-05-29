"""Deploy eval notebooks and create a manual-trigger job in Databricks.

Uploads the notebooks to the workspace and creates a job with no schedule,
ready for manual execution via the Databricks UI or CLI.

Usage:
    python scripts/deploy_notebooks.py

    # Then run the job via CLI:
    databricks jobs run-now --job-id <JOB_ID>
"""
import base64
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    JobSettings,
    Task,
    NotebookTask,
    Source,
)
from databricks.sdk.service.workspace import ImportFormat, Language

# ── Config ────────────────────────────────────────────────────

CATALOG = os.environ.get("CATALOG")
SCHEMA = os.environ.get("SCHEMA", "databrickstv")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/databrickstv-agent")

if not CATALOG:
    print("Error: CATALOG environment variable is not set.")
    print("Set it in .env or export it: export CATALOG=labelbricks_test_catalog")
    sys.exit(1)

WORKSPACE_DIR = "/Shared/projects/databrickstv-agent/notebooks"

NOTEBOOKS = {
    "create_eval_dataset": "notebooks/create_eval_dataset.py",
    "sync_failing_traces": "notebooks/sync_failing_traces.py",
}

JOB_NAME = "databrickstv-sync-failing-traces"


def main():
    w = WorkspaceClient()

    # ── 1. Upload notebooks ──────────────────────────────────
    print(f"Uploading notebooks to {WORKSPACE_DIR}/")

    w.workspace.mkdirs(WORKSPACE_DIR)

    for name, local_path in NOTEBOOKS.items():
        remote_path = f"{WORKSPACE_DIR}/{name}"
        with open(local_path, "rb") as f:
            w.workspace.import_(
                path=remote_path,
                content=base64.b64encode(f.read()).decode("utf-8"),
                format=ImportFormat.SOURCE,
                language=Language.PYTHON,
                overwrite=True,
            )
        print(f"  Uploaded {local_path} -> {remote_path}")

    # ── 2. Create job (no schedule) ──────────────────────────
    print(f"\nCreating job: {JOB_NAME}")

    # Check if job already exists
    existing_jobs = list(w.jobs.list(name=JOB_NAME))
    if existing_jobs:
        job_id = existing_jobs[0].job_id
        w.jobs.reset(
            job_id=job_id,
            new_settings=JobSettings(
                name=JOB_NAME,
                tasks=[
                    Task(
                        task_key="sync_failing_traces",
                        notebook_task=NotebookTask(
                            notebook_path=f"{WORKSPACE_DIR}/sync_failing_traces",
                            base_parameters={
                                "catalog": CATALOG,
                                "schema": SCHEMA,
                                "experiment_name": EXPERIMENT_NAME,
                            },
                            source=Source.WORKSPACE,
                        ),
                    )
                ],
            ),
        )
        print(f"  Updated existing job (ID: {job_id})")
    else:
        job = w.jobs.create(
            name=JOB_NAME,
            tasks=[
                Task(
                    task_key="sync_failing_traces",
                    notebook_task=NotebookTask(
                        notebook_path=f"{WORKSPACE_DIR}/sync_failing_traces",
                        base_parameters={
                            "catalog": CATALOG,
                            "schema": SCHEMA,
                            "experiment_name": EXPERIMENT_NAME,
                        },
                        source=Source.WORKSPACE,
                    ),
                )
            ],
        )
        job_id = job.job_id
        print(f"  Created job (ID: {job_id})")

    # ── 3. Print summary ─────────────────────────────────────
    host = w.config.host.rstrip("/")
    print(f"\n{'=' * 60}")
    print(f"Notebooks: {host}/#workspace{WORKSPACE_DIR}")
    print(f"Job:       {host}/#job/{job_id}")
    print(f"{'=' * 60}")
    print(f"\nTo run the job via CLI:")
    print(f"  databricks jobs run-now {job_id}")


if __name__ == "__main__":
    main()
