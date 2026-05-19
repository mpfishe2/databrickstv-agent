# Databricks notebook source

# MAGIC %md
# MAGIC # Create Eval Dataset
# MAGIC
# MAGIC One-time setup notebook that creates the UC eval dataset and populates it
# MAGIC with the 30 baseline samples. Safe to re-run — it will skip creation if the
# MAGIC dataset already exists and merge records idempotently.
# MAGIC
# MAGIC **Parameters (Databricks widgets):**
# MAGIC - `catalog` — Unity Catalog name (e.g., `labelbricks_test_catalog`)
# MAGIC - `schema` — Schema name (default: `databrickstv`)
# MAGIC - `experiment_name` — MLflow experiment path (default: `/Shared/databrickstv-agent`)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog Name")
dbutils.widgets.text("schema", "databrickstv", "Schema Name")
dbutils.widgets.text("experiment_name", "/Shared/databrickstv-agent", "MLflow Experiment")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")

if not catalog:
    raise ValueError("The 'catalog' widget must be set.")

UC_DATASET = f"{catalog}.{schema}.eval_dataset"
print(f"Catalog:    {catalog}")
print(f"Schema:     {schema}")
print(f"Dataset:    {UC_DATASET}")
print(f"Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define the 30 baseline eval samples
# MAGIC
# MAGIC Each sample has an `inputs` dict with a user `message` and an `expectations`
# MAGIC dict describing which tool(s) the agent should call (and expected verdicts
# MAGIC where applicable).

# COMMAND ----------

eval_data = [
    # recommend_content (8 samples)
    {"inputs": {"message": "What should user U0001 watch next?"}, "expectations": {"expected_tools": ["recommend_content"]}},
    {"inputs": {"message": "Recommend something fun for viewer U0045."}, "expectations": {"expected_tools": ["recommend_content"]}},
    {"inputs": {"message": "Suggest a few titles that U0312 would enjoy based on their history."}, "expectations": {"expected_tools": ["recommend_content"]}},
    {"inputs": {"message": "What's good to watch for user U0888? They like thrillers."}, "expectations": {"expected_tools": ["recommend_content"]}},
    {"inputs": {"message": "Can you pick the next binge for U1500?"}, "expectations": {"expected_tools": ["recommend_content"]}},
    {"inputs": {"message": "I need content recommendations for user U0210, preferably something family-friendly."}, "expectations": {"expected_tools": ["recommend_content"]}},
    {"inputs": {"message": "Give me a personalized watchlist for U1750."}, "expectations": {"expected_tools": ["recommend_content"]}},
    {"inputs": {"message": "What would you recommend for user U9999?"}, "expectations": {"expected_tools": ["recommend_content"]}},

    # check_brand_safety (8 samples)
    {"inputs": {"message": "Is campaign C008 brand-safe for content CT0328?"}, "expectations": {"expected_tools": ["check_brand_safety"], "expected_verdict": "safe"}},
    {"inputs": {"message": "Check brand safety of ad campaign C048 against title CT0347."}, "expectations": {"expected_tools": ["check_brand_safety"], "expected_verdict": "safe"}},
    {"inputs": {"message": "Would it be okay to run campaign C015 alongside content CT0112?"}, "expectations": {"expected_tools": ["check_brand_safety"], "expected_verdict": "safe"}},
    {"inputs": {"message": "Verify that C038 meets brand-safety requirements for CT0230."}, "expectations": {"expected_tools": ["check_brand_safety"], "expected_verdict": "safe"}},
    {"inputs": {"message": "Can we pair campaign C014 with content CT0080? Any brand-safety concerns?"}, "expectations": {"expected_tools": ["check_brand_safety"], "expected_verdict": "unsafe"}},
    {"inputs": {"message": "Run a brand-safety check for C007 on content item CT0444."}, "expectations": {"expected_tools": ["check_brand_safety"], "expected_verdict": "unsafe"}},
    {"inputs": {"message": "Is it safe to show C014 ads during CT0182?"}, "expectations": {"expected_tools": ["check_brand_safety"], "expected_verdict": "unsafe"}},
    {"inputs": {"message": "Evaluate whether campaign C036 is appropriate for content CT0382."}, "expectations": {"expected_tools": ["check_brand_safety"], "expected_verdict": "unsafe"}},

    # explore_data (8 samples)
    {"inputs": {"message": "What are the top 10 most popular content titles?"}, "expectations": {"expected_tools": ["explore_data"]}},
    {"inputs": {"message": "How many users are in each region?"}, "expectations": {"expected_tools": ["explore_data"]}},
    {"inputs": {"message": "What's the average completion percentage by genre?"}, "expectations": {"expected_tools": ["explore_data"]}},
    {"inputs": {"message": "Show me the distribution of subscription tiers across all users."}, "expectations": {"expected_tools": ["explore_data"]}},
    {"inputs": {"message": "Which viewer segments have the highest average ratings given?"}, "expectations": {"expected_tools": ["explore_data"]}},
    {"inputs": {"message": "How many watch events happened per device type last month?"}, "expectations": {"expected_tools": ["explore_data"]}},
    {"inputs": {"message": "List all active ad campaigns and the number of content reviews each one has."}, "expectations": {"expected_tools": ["explore_data"]}},
    {"inputs": {"message": "What percentage of content in the catalog is rated TV-MA?"}, "expectations": {"expected_tools": ["explore_data"]}},

    # log_feedback (3 samples)
    {"inputs": {"message": "That recommendation for U0042 was spot on, loved it!"}, "expectations": {"expected_tools": ["log_feedback"]}},
    {"inputs": {"message": "The suggestions for U0312 were terrible -- none of them matched their taste."}, "expectations": {"expected_tools": ["log_feedback"]}},
    {"inputs": {"message": "Actually, U0888 prefers documentaries, not horror. Please note that for next time."}, "expectations": {"expected_tools": ["log_feedback"]}},

    # multi-tool (3 samples)
    {"inputs": {"message": "Recommend content for U0042 and then check if campaign C008 is brand-safe for the top recommendation."}, "expectations": {"expected_tools": ["recommend_content", "check_brand_safety"]}},
    {"inputs": {"message": "First, tell me which genre is most watched overall, then suggest something in that genre for user U0210."}, "expectations": {"expected_tools": ["explore_data", "recommend_content"]}},
    {"inputs": {"message": "Pull the top 5 content titles by popularity, then verify brand safety of campaign C015 for each of them."}, "expectations": {"expected_tools": ["explore_data", "check_brand_safety"]}},
]

print(f"Loaded {len(eval_data)} eval samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create or get the eval dataset
# MAGIC
# MAGIC Uses `create_dataset` if the table doesn't exist yet, otherwise `get_dataset`.

# COMMAND ----------

import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)

try:
    dataset = mlflow.genai.datasets.create_dataset(name=UC_DATASET)
    print(f"Created new dataset: {UC_DATASET}")
except Exception as e:
    if "TABLE_ALREADY_EXISTS" in str(e):
        dataset = mlflow.genai.datasets.get_dataset(UC_DATASET)
        print(f"Dataset already exists: {UC_DATASET}")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Merge the baseline samples into the dataset
# MAGIC
# MAGIC `merge_records` is idempotent — re-running won't create duplicates.

# COMMAND ----------

dataset.merge_records(eval_data)

df = dataset.to_df()
print(f"Dataset now has {len(df)} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC The eval dataset is ready at `{UC_DATASET}`. It will be used by:
# MAGIC - `tests/eval_benchmark.py` — PR eval gate
# MAGIC - `sync_failing_traces` notebook — feedback loop from production
