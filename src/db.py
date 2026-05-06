"""
Database helpers for querying Delta Lake tables via the
Databricks SQL Statement Execution API.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
from mlflow.entities import SpanType
from databricks.sdk.service.sql import (
    Disposition,
    Format,
    StatementState,
)

from src.config import get_workspace_client, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level execution
# ---------------------------------------------------------------------------


@mlflow.trace(name="execute_sql", span_type=SpanType.RETRIEVER)
def execute_sql(sql: str, *, timeout_seconds: float = 60.0) -> list[dict[str, Any]]:
    """Execute a SQL statement against the configured warehouse and return rows
    as a list of dicts.

    Uses the Databricks SDK ``StatementExecution`` API, which is the
    recommended approach for Databricks Apps.

    Parameters
    ----------
    sql:
        The SQL string to execute.  Should already include fully-qualified
        table references (catalog.schema.table).
    timeout_seconds:
        Maximum wall-clock time to wait for the statement to finish.

    Returns
    -------
    list[dict[str, Any]]
        Each dict maps column name -> value for one row.

    Raises
    ------
    RuntimeError
        If the statement fails or times out.
    """
    w = get_workspace_client()
    logger.info("Executing SQL: %s", sql[:200])

    response = w.statement_execution.execute_statement(
        warehouse_id=settings.warehouse_id,
        statement=sql,
        catalog=settings.catalog,
        schema=settings.schema,
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        wait_timeout="50s",
    )

    # Poll until terminal state
    deadline = time.monotonic() + timeout_seconds
    while response.status and response.status.state in (
        StatementState.PENDING,
        StatementState.RUNNING,
    ):
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"SQL statement timed out after {timeout_seconds}s. "
                f"Statement ID: {response.statement_id}"
            )
        time.sleep(0.5)
        response = w.statement_execution.get_statement(response.statement_id)

    # Check for failure
    if response.status and response.status.state == StatementState.FAILED:
        error_msg = ""
        if response.status.error:
            error_msg = response.status.error.message or str(response.status.error)
        raise RuntimeError(f"SQL statement failed: {error_msg}\nSQL: {sql}")

    # Parse results
    columns = []
    if response.manifest and response.manifest.schema and response.manifest.schema.columns:
        columns = [col.name for col in response.manifest.schema.columns]

    rows: list[dict[str, Any]] = []
    if response.result and response.result.data_array:
        for row_arr in response.result.data_array:
            row_dict = {}
            for i, col_name in enumerate(columns):
                val = row_arr[i] if i < len(row_arr) else None
                row_dict[col_name] = val
            rows.append(row_dict)

    logger.info("SQL returned %d rows", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def table(name: str) -> str:
    """Return a fully-qualified table reference."""
    return f"{settings.fqn}.{name}"


def query_table(
    table_name: str,
    columns: str = "*",
    where: str = "",
    order_by: str = "",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build and execute a simple SELECT against a DatabricksTV table.

    Parameters
    ----------
    table_name:
        Unqualified table name (e.g. ``"content_catalog"``).
    columns:
        Comma-separated column list or ``"*"``.
    where:
        Optional WHERE clause **without** the ``WHERE`` keyword.
    order_by:
        Optional ORDER BY clause **without** the ``ORDER BY`` keyword.
    limit:
        Optional row limit.
    """
    parts = [f"SELECT {columns} FROM {table(table_name)}"]
    if where:
        parts.append(f"WHERE {where}")
    if order_by:
        parts.append(f"ORDER BY {order_by}")
    if limit is not None:
        parts.append(f"LIMIT {limit}")
    return execute_sql(" ".join(parts))


def format_rows_as_text(
    rows: list[dict[str, Any]],
    *,
    max_rows: int = 50,
) -> str:
    """Format query results into a human-readable text block.

    Falls back to JSON when the data is complex.
    """
    if not rows:
        return "(no results)"

    display = rows[:max_rows]

    # Try a simple tabular format
    cols = list(display[0].keys())
    lines = [" | ".join(cols)]
    lines.append("-" * len(lines[0]))
    for r in display:
        lines.append(" | ".join(str(r.get(c, "")) for c in cols))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)
