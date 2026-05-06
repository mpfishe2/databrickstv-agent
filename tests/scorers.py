"""Scorer / judge definitions for the DatabricksTV Agent app.

Shared by offline evaluation (eval_runner.py) and production monitoring
(register_monitors.py). Keep all scorer definitions here so both paths
evaluate with identical criteria.
"""
from mlflow.genai.scorers import Guidelines, Safety, RelevanceToQuery, scorer
from mlflow.entities import Feedback, Trace, SpanType

# ── Built-in scorers ──────────────────────────────────────────────────────────

safety = Safety()
relevance = RelevanceToQuery()

# ── Guidelines-based scorers ─────────────────────────────────────────────────

agent_quality = Guidelines(
    name="agent_quality",
    guidelines=[
        "The response must directly address the user's question",
        "The response must be concise and well-formatted with markdown when appropriate",
        "If recommending content, the response must include title, genre, and reasoning for each recommendation",
        "If checking brand safety, the response must include a clear SAFE or UNSAFE verdict with detailed reasoning",
        "If answering a data question, the response must reference the actual query results with numbers",
        "The response must not fabricate content titles, campaign IDs, user IDs, or statistics",
    ],
)

brand_safety_quality = Guidelines(
    name="brand_safety_quality",
    guidelines=[
        "The response must include both the campaign name and content title",
        "The response must list the campaign's safety requirements",
        "The response must list the content's content warnings",
        "The response must provide a clear SAFE or UNSAFE verdict",
        "If UNSAFE, the response must enumerate each specific conflict between requirements and warnings",
        "The response must mention any prior human review if one exists on file",
    ],
)

# ── Custom trace-based scorers ────────────────────────────────────────────────


@scorer
def correct_tool_called(
    inputs: dict, outputs: dict, expectations: dict, trace: Trace = None
) -> Feedback:
    """Verify that the agent called the expected tool(s) for the given query."""
    expected_tools = expectations.get("expected_tools", [])
    if trace is None:
        return Feedback(name="correct_tool_called", value="skipped", rationale="No trace available")
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    called_tools = [s.name for s in tool_spans]
    all_found = all(t in called_tools for t in expected_tools)
    return Feedback(
        name="correct_tool_called",
        value="yes" if all_found else "no",
        rationale=(
            f"Expected tools: {expected_tools}. "
            f"Called tools: {called_tools}."
        ),
    )


@scorer
def brand_safety_verdict_correct(
    inputs: dict, outputs: dict, expectations: dict, trace: Trace = None
) -> Feedback:
    """Verify the brand-safety verdict matches the expected outcome."""
    expected_verdict = expectations.get("expected_verdict", "").lower()

    if not expected_verdict:
        return Feedback(name="brand_safety_verdict_correct", value="skipped", rationale="No expected verdict")

    # Look for the check_brand_safety tool span or any span whose output
    # contains the word VERDICT.
    if trace is None:
        # Fall back to checking the output text directly
        verdict_text = str(outputs)
        extracted = ""
        upper_text = verdict_text.upper()
        if "UNSAFE" in upper_text:
            extracted = "unsafe"
        elif "SAFE" in upper_text:
            extracted = "safe"
        match = extracted == expected_verdict
        return Feedback(
            name="brand_safety_verdict_correct",
            value="yes" if match else "no",
            rationale=f"Expected: {expected_verdict}. Extracted from output: {extracted or 'none found'}.",
        )

    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    verdict_text = ""
    for span in tool_spans:
        if span.name == "check_brand_safety":
            verdict_text = str(getattr(span, "outputs", ""))
            break
        span_output = str(getattr(span, "outputs", ""))
        if "VERDICT" in span_output.upper():
            verdict_text = span_output
            break

    # If we didn't find it in tool spans, fall back to the final output.
    if not verdict_text:
        verdict_text = str(outputs)

    extracted = ""
    upper_text = verdict_text.upper()
    if "UNSAFE" in upper_text:
        extracted = "unsafe"
    elif "SAFE" in upper_text:
        extracted = "safe"

    match = extracted == expected_verdict
    return Feedback(
        name="brand_safety_verdict_correct",
        value="yes" if match else "no",
        rationale=(
            f"Expected verdict: {expected_verdict}. "
            f"Extracted verdict: {extracted or 'none found'}."
        ),
    )


@scorer
def llm_latency_check(trace: Trace = None) -> list[Feedback]:
    """Check total LLM call latency and flag if over threshold."""
    if trace is None:
        return [Feedback(name="total_llm_latency_ms", value=0, rationale="No trace available"),
                Feedback(name="llm_under_30s", value="skipped", rationale="No trace available")]
    llm_spans = trace.search_spans(span_type=SpanType.LLM)
    total_ms = sum(
        (s.end_time_ns - s.start_time_ns) / 1e6 for s in llm_spans
    )
    return [
        Feedback(
            name="total_llm_latency_ms",
            value=round(total_ms, 2),
            rationale=f"{len(llm_spans)} LLM call(s), total {total_ms:.0f}ms",
        ),
        Feedback(
            name="llm_under_30s",
            value="yes" if total_ms < 30_000 else "no",
            rationale=f"LLM total {total_ms:.0f}ms vs 30s threshold",
        ),
    ]
