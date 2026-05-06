"""
DatabricksTV recommendation agent with native tool-calling loop and MLflow tracing.

Uses a Databricks Foundation Model API endpoint as the LLM backend
and dispatches four tools for content recommendation, brand safety,
data exploration, and feedback logging.
"""

from __future__ import annotations

import json
import logging

import mlflow
import requests
from mlflow.entities import SpanType

from src.config import get_oauth_token, get_workspace_host, settings
from src.tools import (
    recommend_content,
    check_brand_safety,
    explore_data,
    log_feedback,
    SCHEMA_DESCRIPTION,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are the DatabricksTV Recommendation Agent -- an AI assistant
for a fictional streaming platform called DatabricksTV.

You help media and entertainment professionals with:
1. **Content Recommendations** -- suggest what viewers should watch next
2. **Brand Safety** -- check if ad campaigns are safe to pair with content
3. **Data Exploration** -- answer analytical questions about viewers, content, and ads
4. **Feedback** -- capture feedback on recommendations

IMPORTANT GUIDELINES:
- Always be helpful, concise, and professional.
- When recommending content, always use the recommend_content tool with the user's ID.
- When checking brand safety, always use the check_brand_safety tool.
- For data questions, generate valid Databricks SQL and use the explore_data tool.
  The SQL must use fully-qualified table names.
- When the user gives feedback, use the log_feedback tool.
- If no user ID is provided and one is needed, ask for it.
- Format responses clearly with markdown when helpful.
- You can handle follow-up questions about previous results.

{SCHEMA_DESCRIPTION}
"""


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "recommend_content",
        "description": "Recommend the top 5 unwatched content titles for a DatabricksTV viewer based on their audience segment preferences and watch history.",
        "parameters": {"type": "object", "properties": {
            "user_id": {"type": "string", "description": "The viewer's user ID, e.g. 'U0042'. Format: U followed by 4 digits (U0001 through U2000)."}
        }, "required": ["user_id"]},
    }},
    {"type": "function", "function": {
        "name": "check_brand_safety",
        "description": "Check whether an advertising campaign is brand-safe for a specific piece of content on DatabricksTV.",
        "parameters": {"type": "object", "properties": {
            "campaign_id": {"type": "string", "description": "The ad campaign ID, e.g. 'C008'. Format: C followed by 3 digits (C001 through C050)."},
            "content_id": {"type": "string", "description": "The content ID, e.g. 'CT0123'. Format: CT followed by 4 digits (CT0001 through CT0500)."}
        }, "required": ["campaign_id", "content_id"]},
    }},
    {"type": "function", "function": {
        "name": "explore_data",
        "description": "Execute a SQL query against the DatabricksTV data warehouse. YOU must generate the SQL yourself using schema knowledge. Pass a complete SQL SELECT statement.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "A valid Databricks SQL SELECT statement using fully-qualified table names."}
        }, "required": ["question"]},
    }},
    {"type": "function", "function": {
        "name": "log_feedback",
        "description": "Log user feedback on a recommendation or brand-safety check result.",
        "parameters": {"type": "object", "properties": {
            "recommendation_context": {"type": "string", "description": "What was being evaluated"},
            "feedback_type": {"type": "string", "description": "One of: positive, negative, neutral, correction"},
            "comment": {"type": "string", "description": "The user's feedback"}
        }, "required": ["recommendation_context", "feedback_type", "comment"]},
    }},
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

TOOL_DISPATCH = {
    "recommend_content": lambda args: recommend_content(args["user_id"]),
    "check_brand_safety": lambda args: check_brand_safety(args["campaign_id"], args["content_id"]),
    "explore_data": lambda args: explore_data(args["question"]),
    "log_feedback": lambda args: log_feedback(args["recommendation_context"], args["feedback_type"], args["comment"]),
}


@mlflow.trace(name="execute_tool", span_type=SpanType.TOOL)
def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call and return the result as a string."""
    handler = TOOL_DISPATCH.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = handler(arguments)
        return result  # tool functions already return strings
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

@mlflow.trace(name="call_llm", span_type=SpanType.LLM)
def call_llm(messages: list) -> dict:
    """Call Foundation Model API, return the parsed response JSON."""
    token = get_oauth_token()
    host = get_workspace_host()
    url = f"{host}/serving-endpoints/{settings.serving_endpoint}/invocations"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"messages": messages, "tools": TOOLS, "max_tokens": 4096, "temperature": 0.7},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@mlflow.trace(name="run_agent", span_type=SpanType.CHAIN)
def run_agent(messages: list) -> tuple[str, list[dict]]:
    """Run the agent loop (up to 5 tool-call iterations).

    Returns (final_response_text, tool_trace) where tool_trace is a list of
    {tool, args, result} dicts for the UI to display.
    """
    tool_trace = []

    for _ in range(5):
        result = call_llm(messages)
        choice = result["choices"][0]
        msg = choice["message"]

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", ""), tool_trace

        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            tool_result = execute_tool(fn_name, fn_args)
            tool_trace.append({"tool": fn_name, "args": fn_args, "result": tool_result})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

    return messages[-1].get("content", ""), tool_trace
