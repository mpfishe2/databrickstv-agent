-- ============================================================================
-- Unity Catalog function registrations for the DatabricksTV agent
--
-- BEFORE RUNNING: Find and replace the catalog/schema below to match your
-- environment. The defaults use placeholder values:
--   Catalog: YOUR_CATALOG    -> replace with your UC catalog name
--   Schema:  databrickstv    -> replace if you used a different schema
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. check_brand_safety
--    Cross-checks an ad campaign's safety requirements against a content
--    item's warnings. Also surfaces any prior human review verdicts.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION YOUR_CATALOG.databrickstv.check_brand_safety(
  campaign_id STRING,
  content_id STRING
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Checks whether content is brand-safe for a given ad campaign by comparing content warnings against campaign safety requirements and surfacing prior human reviews.'
AS $$
  import json

  # Query campaign
  camp_rows = spark.sql(f"""
    SELECT campaign_id, campaign_name, brand_name, content_safety_requirements
    FROM YOUR_CATALOG.databrickstv.ad_campaigns
    WHERE campaign_id = '{campaign_id}'
  """).collect()
  if not camp_rows:
    return f"Campaign '{campaign_id}' not found."
  camp = camp_rows[0]

  # Query content
  content_rows = spark.sql(f"""
    SELECT content_id, title, genre, rating, content_warnings, ad_tier
    FROM YOUR_CATALOG.databrickstv.content_catalog
    WHERE content_id = '{content_id}'
  """).collect()
  if not content_rows:
    return f"Content '{content_id}' not found."
  ct = content_rows[0]

  # Parse arrays
  safety_reqs = camp["content_safety_requirements"] or []
  warnings = ct["content_warnings"] or []

  # Cross-check: strip "no_" prefix from requirements and compare
  conflicts = []
  for req in safety_reqs:
    prohibited = req.lower().replace("no_", "").replace("-", "_").strip()
    for warn in warnings:
      if prohibited in warn.lower().replace("-", "_"):
        conflicts.append(f"Campaign requires '{req}' but content has warning '{warn}'")

  # Check prior human reviews
  review_rows = spark.sql(f"""
    SELECT is_brand_safe, safety_score, reviewer_notes
    FROM YOUR_CATALOG.databrickstv.content_ad_reviews
    WHERE campaign_id = '{campaign_id}' AND content_id = '{content_id}'
    ORDER BY review_date DESC LIMIT 1
  """).collect()

  # Build verdict
  lines = [
    "=== Brand Safety Verdict ===",
    f"Campaign: {camp['campaign_name']} ({campaign_id}) by {camp['brand_name']}",
    f"Content:  {ct['title']} ({content_id})",
    f"Rating:   {ct['rating']}",
    f"Ad Tier:  {ct['ad_tier'] or 'N/A'}",
    "",
    f"Safety Requirements: {', '.join(safety_reqs) if safety_reqs else 'None'}",
    f"Content Warnings: {', '.join(warnings) if warnings else 'None'}",
    "",
  ]

  if conflicts:
    lines.append("VERDICT: UNSAFE")
    lines.append(f"Conflicts ({len(conflicts)}):")
    for c in conflicts:
      lines.append(f"  - {c}")
  else:
    lines.append("VERDICT: SAFE")
    lines.append("No conflicts detected.")

  if review_rows:
    r = review_rows[0]
    verdict_str = "SAFE" if r["is_brand_safe"] else "UNSAFE"
    lines.extend(["", "Prior Human Review:",
      f"  Verdict: {verdict_str} (score: {r['safety_score']}/100)",
      f"  Notes: {r['reviewer_notes']}"])
  else:
    lines.extend(["", "No prior human review on file."])

  return "\n".join(lines)
$$;

-- ---------------------------------------------------------------------------
-- 2. log_feedback
--    Records user feedback about a recommendation. Currently returns a
--    confirmation string; a future version would INSERT into a feedback table.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION YOUR_CATALOG.databrickstv.log_feedback(
  recommendation_context STRING,
  feedback_type STRING,
  comment STRING
)
RETURNS STRING
COMMENT 'Logs user feedback on a recommendation. Validates feedback_type and returns a confirmation string.'
RETURN
  CASE
    WHEN feedback_type NOT IN ('positive', 'negative', 'neutral', 'correction')
      THEN CONCAT('ERROR: Invalid feedback_type "', feedback_type,
                   '". Must be one of: positive, negative, neutral, correction.')
    ELSE CONCAT(
      'Feedback logged successfully.\n',
      'Type   : ', feedback_type, '\n',
      'Context: ', recommendation_context, '\n',
      'Comment: ', comment
    )
  END;
