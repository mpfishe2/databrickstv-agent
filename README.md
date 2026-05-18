# DatabricksTV Recommendation Agent

A conversational AI agent for **DatabricksTV**, a fictional streaming platform. Built as a Databricks App with FastAPI, MLflow tracing, and Claude as the LLM backend.

The agent helps media and entertainment professionals with content recommendations, brand safety checks, data exploration, and feedback capture — all powered by tool-calling against Delta Lake tables.

## Architecture

```
User ──> FastAPI (src/app.py)
            |
            v
         Agent Loop (src/agent.py)  <-->  Claude (Foundation Model API)
            |
            v
         Tool Dispatch
            |-- recommend_content    -> users, viewer_segments, watch_history, content_catalog
            |-- check_brand_safety   -> ad_campaigns, content_catalog, content_ad_reviews
            |-- explore_data         -> Any table via SQL
            +-- log_feedback         -> stdout (future: Lakebase)
            |
            v
         Databricks SQL Warehouse (src/db.py)
```

All tool calls are traced with MLflow and stored in Unity Catalog for evaluation and production monitoring.

## Getting Started

### Prerequisites

- **Databricks workspace** with Unity Catalog enabled
- **SQL warehouse** provisioned ([how to create one](https://docs.databricks.com/en/compute/sql-warehouse/create.html))
- **Databricks CLI** installed and authenticated (`pip install databricks-cli && databricks auth login`)
- **Foundation Model API** endpoint available (default: `databricks-claude-sonnet-4-6`)
- **Python 3.12+**

### Configuration Reference

Every value you need to configure, where to set it, and how to find it:

| Value | Where to Set | How to Find It | Required? |
|-------|-------------|----------------|-----------|
| `DATABRICKS_PROFILE` | `.env` | The profile name you chose during `databricks auth login` | Yes |
| `CATALOG` | `.env` + `databricks.yml` | Unity Catalog explorer in workspace sidebar | Yes |
| `WAREHOUSE_ID` | `.env` + GitHub secret | SQL Warehouses page -> click warehouse -> ID in the URL bar | Yes |
| `SCHEMA` | `.env` (optional) | Defaults to `databrickstv`. Only change if you used a different schema name for data tables | No |
| `SERVING_ENDPOINT` | `.env` (optional) | Serving endpoints page in workspace. Default: `databricks-claude-sonnet-4-6` | No |
| `MLFLOW_EXPERIMENT_NAME` | `.env` (optional) | Default: `/Shared/databrickstv-agent`. Change if you want a different experiment path | No |
| `DATABRICKS_HOST` | GitHub repo secret | Your workspace URL (e.g., `https://xxx.cloud.databricks.com`) | CI/CD only |
| `DATABRICKS_TOKEN` | GitHub repo secret | Settings -> Developer -> Access Tokens in workspace | CI/CD only |
| `CATALOG_NAME` | GitHub repo variable | Same as `CATALOG` — used by bundle deploy in CI | CI/CD only |

### Step-by-Step Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd databrickstv-agent

# 2. Configure environment
cp .env.example .env
# Edit .env — fill in DATABRICKS_PROFILE, CATALOG, and WAREHOUSE_ID

# 3. Install dependencies
pip install -r requirements.txt

# 4. Load your .env (or use direnv/dotenv)
export $(grep -v '^#' .env | xargs)

# 5. Seed the data tables
# Run the data generation notebook in your workspace, or use:
#   python -m setup.seed_data  (if available)

# 6. Create MLflow experiment and bind to UC trace storage
python -m setup.setup_experiment
python -m setup.setup_uc_traces

# 7. Start the app locally
python -m src.app
# Chat UI available at http://localhost:8000
```

### CI/CD Setup (GitHub Actions)

The project includes two GitHub Actions workflows:
- **`eval_benchmark.yml`** — On every PR: deploys to staging, runs eval, posts results as PR comment
- **`deploy.yml`** — On merge to main: deploys to production

To enable CI/CD:

1. **Set GitHub repository secrets** (Settings -> Secrets and variables -> Actions -> Secrets):
   - `DATABRICKS_HOST` — your workspace URL (e.g., `https://xxx.cloud.databricks.com`)
   - `DATABRICKS_TOKEN` — a personal access token or service principal token
   - `WAREHOUSE_ID` — your SQL warehouse ID

2. **Set GitHub repository variables** (Settings -> Secrets and variables -> Actions -> Variables):
   - `CATALOG_NAME` — your Unity Catalog name (used by `databricks bundle deploy` via `BUNDLE_VAR_catalog_name`)

3. **Self-hosted runner** — The workflows use `runs-on: self-hosted`. Set up a [GitHub Actions self-hosted runner](https://docs.github.com/en/actions/hosting-your-own-runners) or change to `runs-on: ubuntu-latest` for GitHub-hosted runners.

---

## Project Structure

```
databrickstv-agent/
|-- src/                            # Application code
|   |-- app.py                      #   FastAPI server + /api/chat endpoint
|   |-- agent.py                    #   LLM agent loop with tool-calling
|   |-- config.py                   #   Auth, environment detection, settings
|   |-- db.py                       #   SQL execution via Statement Execution API
|   +-- tools.py                    #   4 tool implementations + schema description
|
|-- tests/                          # Evaluation
|   |-- eval_data.py                #   30 test samples across all 4 tools
|   |-- eval_benchmark.py           #   CI eval gate (pytest + MLflow judges)
|   |-- eval_runner.py              #   Offline eval script with deterministic checks
|   +-- scorers.py                  #   LLM judges & trace-based scorers
|
|-- setup/                          # One-time workspace setup scripts
|   |-- setup_experiment.py         #   Create MLflow experiment
|   |-- setup_uc_traces.py          #   Bind experiment to UC trace storage
|   +-- register_monitors.py        #   Register production monitoring scorers
|
|-- static/
|   +-- index.html                  #   Chat UI (dark theme, Databricks branding)
|
|-- .env.example                    #   Configuration template (copy to .env)
|-- databricks.yml                  #   Databricks Asset Bundle config
|-- app.yaml                        #   App runtime config (fallback if not using bundle)
+-- requirements.txt
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

### CI Eval Gate (30 samples)

```bash
# Run the full eval suite (used by GitHub Actions)
pytest tests/eval_benchmark.py -v

# Fast local test with fewer samples
EVAL_SAMPLE_LIMIT=5 pytest tests/eval_benchmark.py -v
```

### Offline Eval

```bash
python -m tests.eval_runner
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

## Deploy

```bash
# Deploy to staging
databricks bundle deploy -t staging

# Deploy to production
databricks bundle deploy -t prod
```

This deploys the app as a Databricks App accessible to all workspace users.
