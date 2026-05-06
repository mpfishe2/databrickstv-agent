"""Evaluation dataset for the DatabricksTV agent -- 30 samples.

Each sample contains an ``inputs`` dict with a natural-language ``message``
and an ``expectations`` dict describing which tool(s) the agent should invoke
(and, where applicable, the expected verdict).

Tables live in labelbricks_test_catalog.databrickstv:
    users              2000 rows   (U0001-U2000)
    viewer_segments      15 rows   (S01-S15)
    content_catalog     500 rows   (CT0001-CT0500)
    watch_history     10000 rows
    ad_campaigns         50 rows   (C001-C050)
    content_ad_reviews  200 rows
    content_rights_corpus 25 rows
"""

eval_data = [
    # ------------------------------------------------------------------ #
    #  recommend_content  (8 samples)                                     #
    # ------------------------------------------------------------------ #
    {
        "inputs": {"message": "What should user U0001 watch next?"},
        "expectations": {"expected_tools": ["recommend_content"]},
    },
    {
        "inputs": {"message": "Recommend something fun for viewer U0045."},
        "expectations": {"expected_tools": ["recommend_content"]},
    },
    {
        "inputs": {"message": "Suggest a few titles that U0312 would enjoy based on their history."},
        "expectations": {"expected_tools": ["recommend_content"]},
    },
    {
        "inputs": {"message": "What's good to watch for user U0888? They like thrillers."},
        "expectations": {"expected_tools": ["recommend_content"]},
    },
    {
        "inputs": {"message": "Can you pick the next binge for U1500?"},
        "expectations": {"expected_tools": ["recommend_content"]},
    },
    {
        "inputs": {"message": "I need content recommendations for user U0210, preferably something family-friendly."},
        "expectations": {"expected_tools": ["recommend_content"]},
    },
    {
        "inputs": {"message": "Give me a personalized watchlist for U1750."},
        "expectations": {"expected_tools": ["recommend_content"]},
    },
    {
        # Edge case: non-existent user
        "inputs": {"message": "What would you recommend for user U9999?"},
        "expectations": {"expected_tools": ["recommend_content"]},
    },

    # ------------------------------------------------------------------ #
    #  check_brand_safety  (8 samples)                                    #
    #  Ground truth from content_ad_reviews table                         #
    # ------------------------------------------------------------------ #
    # Safe pairings (is_brand_safe = true)
    {
        "inputs": {"message": "Is campaign C008 brand-safe for content CT0328?"},
        "expectations": {
            "expected_tools": ["check_brand_safety"],
            "expected_verdict": "safe",
        },
    },
    {
        "inputs": {"message": "Check brand safety of ad campaign C048 against title CT0347."},
        "expectations": {
            "expected_tools": ["check_brand_safety"],
            "expected_verdict": "safe",
        },
    },
    {
        "inputs": {"message": "Would it be okay to run campaign C015 alongside content CT0112?"},
        "expectations": {
            "expected_tools": ["check_brand_safety"],
            "expected_verdict": "safe",
        },
    },
    {
        "inputs": {"message": "Verify that C038 meets brand-safety requirements for CT0230."},
        "expectations": {
            "expected_tools": ["check_brand_safety"],
            "expected_verdict": "safe",
        },
    },
    # Unsafe pairings (is_brand_safe = false)
    {
        "inputs": {"message": "Can we pair campaign C014 with content CT0080? Any brand-safety concerns?"},
        "expectations": {
            "expected_tools": ["check_brand_safety"],
            "expected_verdict": "unsafe",
        },
    },
    {
        "inputs": {"message": "Run a brand-safety check for C007 on content item CT0444."},
        "expectations": {
            "expected_tools": ["check_brand_safety"],
            "expected_verdict": "unsafe",
        },
    },
    {
        "inputs": {"message": "Is it safe to show C014 ads during CT0182?"},
        "expectations": {
            "expected_tools": ["check_brand_safety"],
            "expected_verdict": "unsafe",
        },
    },
    {
        "inputs": {"message": "Evaluate whether campaign C036 is appropriate for content CT0382."},
        "expectations": {
            "expected_tools": ["check_brand_safety"],
            "expected_verdict": "unsafe",
        },
    },

    # ------------------------------------------------------------------ #
    #  explore_data  (8 samples)                                          #
    # ------------------------------------------------------------------ #
    {
        "inputs": {"message": "What are the top 10 most popular content titles?"},
        "expectations": {"expected_tools": ["explore_data"]},
    },
    {
        "inputs": {"message": "How many users are in each region?"},
        "expectations": {"expected_tools": ["explore_data"]},
    },
    {
        "inputs": {"message": "What's the average completion percentage by genre?"},
        "expectations": {"expected_tools": ["explore_data"]},
    },
    {
        "inputs": {"message": "Show me the distribution of subscription tiers across all users."},
        "expectations": {"expected_tools": ["explore_data"]},
    },
    {
        "inputs": {"message": "Which viewer segments have the highest average ratings given?"},
        "expectations": {"expected_tools": ["explore_data"]},
    },
    {
        "inputs": {"message": "How many watch events happened per device type last month?"},
        "expectations": {"expected_tools": ["explore_data"]},
    },
    {
        "inputs": {"message": "List all active ad campaigns and the number of content reviews each one has."},
        "expectations": {"expected_tools": ["explore_data"]},
    },
    {
        "inputs": {"message": "What percentage of content in the catalog is rated TV-MA?"},
        "expectations": {"expected_tools": ["explore_data"]},
    },

    # ------------------------------------------------------------------ #
    #  log_feedback  (3 samples)                                          #
    # ------------------------------------------------------------------ #
    {
        "inputs": {
            "message": "That recommendation for U0042 was spot on, loved it!"
        },
        "expectations": {"expected_tools": ["log_feedback"]},
    },
    {
        "inputs": {
            "message": "The suggestions for U0312 were terrible -- none of them matched their taste."
        },
        "expectations": {"expected_tools": ["log_feedback"]},
    },
    {
        "inputs": {
            "message": "Actually, U0888 prefers documentaries, not horror. Please note that for next time."
        },
        "expectations": {"expected_tools": ["log_feedback"]},
    },

    # ------------------------------------------------------------------ #
    #  multi-tool  (3 samples)                                            #
    # ------------------------------------------------------------------ #
    {
        "inputs": {
            "message": (
                "Recommend content for U0042 and then check if campaign C008 "
                "is brand-safe for the top recommendation."
            )
        },
        "expectations": {
            "expected_tools": ["recommend_content", "check_brand_safety"]
        },
    },
    {
        "inputs": {
            "message": (
                "First, tell me which genre is most watched overall, then "
                "suggest something in that genre for user U0210."
            )
        },
        "expectations": {
            "expected_tools": ["explore_data", "recommend_content"]
        },
    },
    {
        "inputs": {
            "message": (
                "Pull the top 5 content titles by popularity, then verify "
                "brand safety of campaign C015 for each of them."
            )
        },
        "expectations": {
            "expected_tools": ["explore_data", "check_brand_safety"]
        },
    },
]
