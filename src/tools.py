"""
Tool implementations for the DatabricksTV recommendation agent.

Each tool function is a plain Python callable that:
  1. Queries Delta Lake tables via ``db.py``
  2. Returns a human-readable string the LLM can relay to the user

These are registered with the OpenAI Agents SDK in ``agent.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import mlflow
from mlflow.entities import SpanType

from src.db import execute_sql, query_table, table, format_rows_as_text
from src.config import settings

logger = logging.getLogger(__name__)


# ===================================================================
# Tool 1 -- Content Recommendations
# ===================================================================

@mlflow.trace(name="recommend_content", span_type=SpanType.TOOL)
def recommend_content(user_id: str) -> str:
    """Recommend unwatched content for a viewer based on their segment preferences.

    Steps:
        1. Look up the user's segment from the ``users`` table.
        2. Fetch preferred genres from ``viewer_segments``.
        3. Retrieve already-watched content IDs from ``watch_history``.
        4. Query ``content_catalog`` for unwatched titles in preferred genres,
           ranked by ``popularity_score``.
        5. Return the top 5 recommendations with reasoning.

    Parameters
    ----------
    user_id:
        The user identifier, e.g. ``"U0042"``.
    """
    user_id = user_id.strip().upper()
    logger.info("recommend_content called for user_id=%s", user_id)

    # 1. Get user + segment
    user_rows = query_table("users", where=f"user_id = '{user_id}'", limit=1)
    if not user_rows:
        return f"User '{user_id}' not found. Valid IDs: U0001 through U2000."
    user = user_rows[0]
    segment_id = user["segment_id"]

    # 2. Get segment preferences
    seg_rows = query_table(
        "viewer_segments", where=f"segment_id = '{segment_id}'", limit=1
    )
    if not seg_rows:
        return f"Segment '{segment_id}' not found in viewer_segments."
    segment = seg_rows[0]
    segment_name = segment["segment_name"]

    # preferred_genres is stored as an ARRAY; the SQL API returns it as a
    # JSON-encoded string.
    preferred_genres_raw = segment.get("preferred_genres", "[]")
    if isinstance(preferred_genres_raw, str):
        try:
            preferred_genres = json.loads(preferred_genres_raw)
        except json.JSONDecodeError:
            preferred_genres = [preferred_genres_raw]
    elif isinstance(preferred_genres_raw, list):
        preferred_genres = preferred_genres_raw
    else:
        preferred_genres = []

    if not preferred_genres:
        return (
            f"User {user_id} belongs to segment '{segment_name}' "
            "but no preferred genres are defined."
        )

    # 3. Already-watched content IDs
    watched_rows = query_table(
        "watch_history",
        columns="DISTINCT content_id",
        where=f"user_id = '{user_id}'",
    )
    watched_ids = {r["content_id"] for r in watched_rows}

    # 4. Build genre filter
    genre_list = ", ".join(f"'{g}'" for g in preferred_genres)
    watched_filter = ""
    if watched_ids:
        watched_list = ", ".join(f"'{cid}'" for cid in watched_ids)
        watched_filter = f"AND content_id NOT IN ({watched_list})"

    sql = (
        f"SELECT content_id, title, genre, sub_genre, description, "
        f"       popularity_score, rating, runtime_minutes "
        f"FROM {table('content_catalog')} "
        f"WHERE genre IN ({genre_list}) {watched_filter} "
        f"ORDER BY popularity_score DESC "
        f"LIMIT 5"
    )
    recs = execute_sql(sql)

    if not recs:
        return (
            f"No unwatched content found for user {user_id} "
            f"(segment: {segment_name}, preferred genres: {preferred_genres}). "
            "They may have seen everything in their preferred genres!"
        )

    # 5. Format response
    lines = [
        f"Top 5 recommendations for **{user_id}**",
        f"Segment: {segment_name} | Preferred genres: {', '.join(preferred_genres)}",
        f"Already watched: {len(watched_ids)} titles",
        "",
    ]
    for i, r in enumerate(recs, 1):
        lines.append(
            f"{i}. **{r['title']}** ({r['genre']} / {r['sub_genre']})\n"
            f"   Rating: {r['rating']} | Popularity: {r['popularity_score']}/100 "
            f"| {r['runtime_minutes']} min\n"
            f"   {r['description']}\n"
            f"   *Why*: Matches your preferred genre '{r['genre']}', "
            f"ranked by platform popularity."
        )
    return "\n".join(lines)


# ===================================================================
# Tool 2 -- Brand Safety Check
# ===================================================================

@mlflow.trace(name="check_brand_safety", span_type=SpanType.TOOL)
def check_brand_safety(campaign_id: str, content_id: str) -> str:
    """Evaluate whether an ad campaign is brand-safe for a specific piece of content.

    Steps:
        1. Fetch campaign safety requirements from ``ad_campaigns``.
        2. Fetch content rating and warnings from ``content_catalog``.
        3. Cross-check for conflicts.
        4. Check ``content_ad_reviews`` for any prior human review.
        5. Return a verdict with reasoning.

    Parameters
    ----------
    campaign_id:
        The ad campaign identifier, e.g. ``"C008"``.
    content_id:
        The content identifier, e.g. ``"CT0123"``.
    """
    campaign_id = campaign_id.strip().upper()
    content_id = content_id.strip().upper()
    logger.info(
        "check_brand_safety called for campaign=%s, content=%s",
        campaign_id,
        content_id,
    )

    # 1. Campaign info
    campaign_rows = query_table(
        "ad_campaigns", where=f"campaign_id = '{campaign_id}'", limit=1
    )
    if not campaign_rows:
        return f"Campaign '{campaign_id}' not found. Valid IDs: C001 through C050."
    campaign = campaign_rows[0]

    safety_reqs_raw = campaign.get("content_safety_requirements", "[]")
    if isinstance(safety_reqs_raw, str):
        try:
            safety_reqs = json.loads(safety_reqs_raw)
        except json.JSONDecodeError:
            safety_reqs = [safety_reqs_raw]
    elif isinstance(safety_reqs_raw, list):
        safety_reqs = safety_reqs_raw
    else:
        safety_reqs = []

    # 2. Content info
    content_rows = query_table(
        "content_catalog",
        columns="content_id, title, genre, rating, content_warnings, ad_tier",
        where=f"content_id = '{content_id}'",
        limit=1,
    )
    if not content_rows:
        return f"Content '{content_id}' not found. Valid IDs: CT0001 through CT0500."
    content = content_rows[0]

    warnings_raw = content.get("content_warnings", "[]")
    if isinstance(warnings_raw, str):
        try:
            content_warnings = json.loads(warnings_raw)
        except json.JSONDecodeError:
            content_warnings = [warnings_raw]
    elif isinstance(warnings_raw, list):
        content_warnings = warnings_raw
    else:
        content_warnings = []

    # 3. Cross-check: do any content warnings violate campaign requirements?
    #    Safety requirements are typically like "no_violence", "no_drug_use", etc.
    #    Content warnings are like "violence", "drug_use", "mild_language", etc.
    conflicts = []
    for req in safety_reqs:
        # Normalize: "no_violence" -> "violence"
        prohibited = req.lower().replace("no_", "").replace("-", "_").strip()
        for warn in content_warnings:
            if prohibited in warn.lower().replace("-", "_"):
                conflicts.append(
                    f"Campaign requires '{req}' but content has warning '{warn}'"
                )

    # 4. Check for prior human reviews
    review_rows = query_table(
        "content_ad_reviews",
        where=f"campaign_id = '{campaign_id}' AND content_id = '{content_id}'",
        order_by="review_date DESC",
        limit=1,
    )

    # 5. Build verdict
    lines = [
        f"Brand Safety Report",
        f"{'=' * 50}",
        f"Campaign: {campaign['campaign_name']} ({campaign_id}) "
        f"by {campaign['brand_name']}",
        f"Content:  {content['title']} ({content_id})",
        f"Rating:   {content['rating']}",
        f"Ad Tier:  {content.get('ad_tier', 'N/A')}",
        "",
        f"Campaign Safety Requirements: {', '.join(safety_reqs) if safety_reqs else 'None specified'}",
        f"Content Warnings: {', '.join(content_warnings) if content_warnings else 'None'}",
        "",
    ]

    if conflicts:
        lines.append("VERDICT: UNSAFE")
        lines.append(f"Conflicts found ({len(conflicts)}):")
        for c in conflicts:
            lines.append(f"  - {c}")
    else:
        lines.append("VERDICT: SAFE")
        lines.append("No conflicts detected between campaign requirements and content warnings.")

    if review_rows:
        review = review_rows[0]
        lines.extend([
            "",
            "Prior Human Review:",
            f"  Brand Safe: {review['is_brand_safe']}",
            f"  Safety Score: {review['safety_score']}/100",
            f"  Relevance Score: {review['relevance_score']}/100",
            f"  Notes: {review['reviewer_notes']}",
            f"  Date: {review['review_date']}",
        ])
    else:
        lines.extend(["", "No prior human review on file for this pairing."])

    return "\n".join(lines)


# ===================================================================
# Tool 3 -- Natural Language Data Exploration
# ===================================================================

# Schema description provided to the LLM for SQL generation.
SCHEMA_DESCRIPTION = f"""
You have access to the following tables in {settings.fqn}:

1. content_catalog (500 rows)
   - content_id STRING (PK, e.g. CT0001)
   - title STRING
   - genre STRING
   - sub_genre STRING
   - rating STRING (e.g. TV-Y, TV-G, TV-PG, TV-14, TV-MA)
   - content_warnings ARRAY<STRING>
   - release_year INT
   - runtime_minutes INT
   - description STRING
   - is_original BOOLEAN
   - popularity_score INT (0-100)
   - ad_tier STRING (premium, standard, free)

2. viewer_segments (15 rows)
   - segment_id STRING (PK, e.g. S01)
   - segment_name STRING
   - age_range STRING
   - avg_watch_hours_weekly DOUBLE
   - preferred_genres ARRAY<STRING>
   - peak_viewing_time STRING
   - ad_tolerance STRING (low, medium, high)
   - churn_risk STRING (low, medium, high)
   - size_millions DOUBLE

3. users (2000 rows)
   - user_id STRING (PK, e.g. U0001)
   - segment_id STRING (FK -> viewer_segments)
   - signup_date DATE
   - subscription_tier STRING (premium_no_ads, standard_with_ads, basic_with_ads)
   - age INT
   - region STRING (US, EU, APAC, LATAM)

4. watch_history (10000 rows)
   - event_id STRING (PK)
   - user_id STRING (FK -> users)
   - content_id STRING (FK -> content_catalog)
   - watch_date DATE
   - completion_pct DOUBLE (0.0-1.0)
   - rating_given INT (1-5, nullable)
   - device STRING

5. ad_campaigns (50 rows)
   - campaign_id STRING (PK, e.g. C008)
   - brand_name STRING
   - industry STRING
   - campaign_name STRING
   - budget_tier STRING
   - daily_impression_target INT
   - content_safety_requirements ARRAY<STRING>
   - target_segments ARRAY<STRING>
   - max_cpm DOUBLE
   - start_date DATE
   - end_date DATE
   - status STRING (active, paused, completed)

6. content_ad_reviews (200 rows)
   - review_id STRING (PK)
   - content_id STRING (FK -> content_catalog)
   - campaign_id STRING (FK -> ad_campaigns)
   - is_brand_safe BOOLEAN
   - safety_score INT (0-100)
   - relevance_score INT (0-100)
   - reviewer_notes STRING
   - review_date DATE
   - reviewer_id STRING

7. content_rights_corpus (25 rows)
   - doc_id STRING (PK)
   - doc_title STRING
   - doc_category STRING
   - content STRING (long text, licensing/policy documents)

IMPORTANT:
- Always use fully qualified table names: {settings.fqn}.<table_name>
- ARRAY columns need special handling: use array_contains() or explode()
- Dates are DATE type, use date functions like date_sub(), current_date(), etc.
- Return at most 50 rows unless the user asks for more.
"""


@mlflow.trace(name="explore_data", span_type=SpanType.TOOL)
def explore_data(question: str) -> str:
    """Answer a natural-language question about DatabricksTV data by generating
    and executing SQL.

    This tool is called by the agent when the user asks an analytical or
    exploratory question.  The *agent itself* generates the SQL (using
    schema knowledge injected into its system prompt).  This function
    simply executes whatever SQL the agent provides and formats the results.

    Parameters
    ----------
    question:
        A SQL query to execute against the DatabricksTV data.
        The agent should generate valid Databricks SQL.
    """
    logger.info("explore_data called with: %s", question[:200])

    # The `question` parameter actually contains the SQL that the agent
    # generated. Execute it directly.
    sql = question.strip()

    # Basic safety: only allow SELECT statements
    if not sql.upper().startswith("SELECT"):
        return (
            "Only SELECT statements are allowed for data exploration. "
            "Please rephrase as a read-only query."
        )

    try:
        rows = execute_sql(sql)
    except RuntimeError as exc:
        return f"SQL execution error: {exc}"

    if not rows:
        return f"Query returned no results.\n\nSQL executed:\n```sql\n{sql}\n```"

    formatted = format_rows_as_text(rows)
    return f"Query results ({len(rows)} rows):\n\n{formatted}\n\nSQL executed:\n```sql\n{sql}\n```"


# ===================================================================
# Tool 4 -- Feedback Logging
# ===================================================================

@mlflow.trace(name="log_feedback", span_type=SpanType.TOOL)
def log_feedback(
    recommendation_context: str,
    feedback_type: str,
    comment: str,
) -> str:
    """Log user feedback on a recommendation or brand-safety check.

    Currently writes to stdout (structured log).  A future iteration will
    persist this to a Lakebase table.

    Parameters
    ----------
    recommendation_context:
        What was recommended or checked (e.g. "content recs for U0042"
        or "brand safety AC012 + C0321").
    feedback_type:
        One of: ``"positive"``, ``"negative"``, ``"neutral"``, ``"correction"``.
    comment:
        Free-text feedback from the user.
    """
    feedback_type = feedback_type.strip().lower()
    valid_types = {"positive", "negative", "neutral", "correction"}
    if feedback_type not in valid_types:
        feedback_type = "neutral"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": recommendation_context,
        "type": feedback_type,
        "comment": comment,
    }
    logger.info("FEEDBACK_LOG: %s", json.dumps(entry))
    print(f"[FEEDBACK] {json.dumps(entry)}")

    return (
        f"Feedback logged successfully.\n"
        f"  Type: {feedback_type}\n"
        f"  Context: {recommendation_context}\n"
        f"  Comment: {comment}\n\n"
        f"Thank you for your feedback! This will be stored in Lakebase "
        f"in a future release."
    )
