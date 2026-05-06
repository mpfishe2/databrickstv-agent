# DatabricksTV Recommendation Agent

A conversational AI agent for **DatabricksTV**, a fictional streaming platform. Built as a Databricks App with FastAPI, MLflow tracing, and Claude as the LLM backend.

The agent helps media and entertainment professionals with content recommendations, brand safety checks, data exploration, and feedback capture — all powered by tool-calling against Delta Lake tables.

## Architecture

```
User ──▶ FastAPI (src/app.py)
            │
            ▼
         Agent Loop (src/agent.py)  ◀──▶  Claude (Foundation Model API)
            │
            ▼
         Tool Dispatch
            ├── recommend_content    → users, viewer_segments, watch_history, content_catalog
            ├── check_brand_safety   → ad_campaigns, content_catalog, content_ad_reviews
            ├── explore_data         → Any table via SQL
            └── log_feedback         → stdout (future: Lakebase)
            │
            ▼
         Databricks SQL Warehouse (src/db.py)
```

All tool calls are traced with MLflow and stored in Unity Catalog for evaluation and production monitoring.

## Project Structure

```
databrickstv-agent/
├── src/                            # Application code
│   ├── app.py                      #   FastAPI server + /api/chat endpoint
│   ├── agent.py                    #   LLM agent loop with tool-calling
│   ├── config.py                   #   Auth, environment detection, settings
│   ├── db.py                       #   SQL execution via Statement Execution API
│   └── tools.py                    #   4 tool implementations + schema description
│
├── tests/                          # Evaluation
│   ├── eval_data.py                #   30 test samples across all 4 tools
│   ├── eval_runner.py              #   Offline eval script with deterministic checks
│   └── scorers.py                  #   LLM judges & trace-based scorers
│
├── setup/                          # One-time workspace setup scripts
│   ├── setup_experiment.py         #   Create MLflow experiment
│   ├── setup_uc_traces.py          #   Bind experiment to UC trace storage
│   └── register_monitors.py        #   Register production monitoring scorers
│
├── scripts/
│   └── register_uc_functions.sql   #   UC function registrations
│
├── static/
│   └── index.html                  #   Chat UI (dark theme, Databricks branding)
│
├── databricks.yml                  #   Databricks Asset Bundle config
├── app.yaml                        #   App runtime config (uvicorn, env vars)
└── requirements.txt
```

## Tools

| Tool | Description | Tables Used |
|------|-------------|-------------|
| `recommend_content` | Top 5 unwatched titles based on viewer segment preferences | users, viewer_segments, watch_history, content_catalog |
| `check_brand_safety` | Verify ad campaign safety against content warnings | ad_campaigns, content_catalog, content_ad_reviews |
| `explore_data` | Execute agent-generated SQL for analytical questions | Any table (SELECT only) |
| `log_feedback` | Capture user feedback on recommendations | stdout |

## Data

- **users** (2,000 rows) — viewer profiles with segments and subscription tiers
- **viewer_segments** (15 rows) — audience segments with preferred genres
- **content_catalog** (500 rows) — titles with ratings, warnings, popularity
- **watch_history** (10,000 rows) — viewing events with completion and ratings
- **ad_campaigns** (50 rows) — campaigns with safety requirements and targeting
- **content_ad_reviews** (200 rows) — human brand safety reviews
- **content_rights_corpus** (25 rows) — licensing and policy documents

## Evaluation

### Offline Eval (30 samples)

```bash
DATABRICKS_CONFIG_PROFILE=my-workspace-test python -m tests.eval_runner
```

Runs all 30 test cases and scores:
- **Tool call correctness** — did the agent call the right tool(s)?
- **Brand safety verdict accuracy** — SAFE/UNSAFE matches expected outcome?

### Scorers / Judges

Defined in `tests/scorers.py`, shared across offline eval and production monitoring:

| Scorer | Type | What it checks |
|--------|------|----------------|
| `Safety` | Built-in | Response safety |
| `RelevanceToQuery` | Built-in | Response relevance to user question |
| `agent_quality` | Guidelines (6 rules) | Conciseness, formatting, no fabrication |
| `brand_safety_quality` | Guidelines (6 rules) | Verdict clarity, conflict enumeration |
| `correct_tool_called` | Custom trace-based | Expected tools were invoked |
| `brand_safety_verdict_correct` | Custom trace-based | SAFE/UNSAFE matches ground truth |
| `llm_latency_check` | Custom trace-based | Total LLM latency under 30s |

### Production Monitoring

Registered via `setup/register_monitors.py`:
- `safety` — 100% sample rate
- `agent_quality` — 25% sample rate
- `brand_safety_quality` — 50% sample rate

## Setup

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- SQL warehouse provisioned
- CLI profile configured (e.g., `my-workpsace-test`)
- Foundation Model API endpoint (`databricks-claude-sonnet-4-6`)

### First-time Setup

```bash
# 1. Create the MLflow experiment
DATABRICKS_CONFIG_PROFILE=my-workpsace-test python -m setup.setup_experiment

# 2. Bind to Unity Catalog trace storage
DATABRICKS_CONFIG_PROFILE=my-workpsace-test python -m setup.setup_uc_traces

# 3. Register production monitors
DATABRICKS_CONFIG_PROFILE=my-workpsace-test python -m setup.register_monitors
```

### Local Development

```bash
pip install -r requirements.txt
DATABRICKS_CONFIG_PROFILE=my-workpsace-test python -m src.app
```

The chat UI will be available at `http://localhost:8000`.

### Deploy to Databricks

```bash
databricks bundle deploy -t fevm
```

This deploys the app as a Databricks App accessible to all workspace users.
