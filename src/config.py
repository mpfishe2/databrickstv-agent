"""
Configuration for the DatabricksTV recommendation agent app.

Handles environment detection, authentication, and workspace settings.
"""

import os
from dataclasses import dataclass, field
from databricks.sdk import WorkspaceClient


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

IS_DATABRICKS_APP: bool = bool(os.environ.get("DATABRICKS_APP_NAME"))


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

def get_workspace_client() -> WorkspaceClient:
    """Return a WorkspaceClient using the appropriate auth method.

    Inside a Databricks App the service-principal credentials are
    injected automatically.  Locally we fall back to the CLI profile.
    """
    if IS_DATABRICKS_APP:
        return WorkspaceClient()
    profile = os.environ.get("DATABRICKS_PROFILE", "fevm-labelbricks-test")
    return WorkspaceClient(profile=profile)


def get_oauth_token() -> str:
    """Obtain a Bearer token for REST / SQL calls.

    Works both inside a Databricks App (service principal) and locally
    (CLI / U2M auth via ``databricks auth login``).
    """
    w = get_workspace_client()
    # Prefer the static token if present (PAT / SP secret).
    if w.config.token:
        return w.config.token
    # Otherwise use the SDK's authenticate() helper which handles
    # OAuth / U2M flows and returns {'Authorization': 'Bearer <tok>'}.
    headers = w.config.authenticate()
    if headers and "Authorization" in headers:
        return headers["Authorization"].replace("Bearer ", "")
    raise RuntimeError("Unable to obtain a Databricks OAuth token.")


def get_workspace_host() -> str:
    """Return the workspace URL with ``https://`` prefix.

    Inside a Databricks App, ``DATABRICKS_HOST`` is the bare hostname
    (no scheme).  The SDK on the other hand includes the scheme.
    """
    if IS_DATABRICKS_APP:
        host = os.environ.get("DATABRICKS_HOST", "")
        if host and not host.startswith("http"):
            return f"https://{host}"
        return host
    w = get_workspace_client()
    return w.config.host.rstrip("/")


# ---------------------------------------------------------------------------
# App-level settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """Immutable application settings pulled from the environment."""

    warehouse_id: str = field(
        default_factory=lambda: os.environ.get("WAREHOUSE_ID", os.getenv("WAREHOUSE_ID"))
    )
    catalog: str = field(
        default_factory=lambda: os.environ.get("CATALOG", "labelbricks_test_catalog")
    )
    schema: str = field(
        default_factory=lambda: os.environ.get("SCHEMA", "databrickstv")
    )
    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_NAME", "databricks-meta-llama-3-3-70b-instruct"
        )
    )
    serving_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "SERVING_ENDPOINT", "databricks-claude-sonnet-4-6"
        )
    )

    @property
    def fqn(self) -> str:
        """Fully-qualified schema reference for SQL queries."""
        return f"{self.catalog}.{self.schema}"


settings = Settings()
