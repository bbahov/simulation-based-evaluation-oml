# active_evaluation.py
#
# Active Evaluation Environment
# =============================
# The model selects which item to recommend at each step and receives
# a reward based on its own choice. This creates a feedback loop:
# the model's actions determine the data it learns from.
#
# Run locally:   python active_evaluation.py
# Run in Colab:  paste into a cell (add `!pip install river` in a prior cell)

import os
import random
import pandas as pd
from river import reco, optim


# ──────────────────────────────────────────────────────────────
# CONFIGURATION  (identical across active and passive scripts)
# ──────────────────────────────────────────────────────────────

USERS = ["Tom", "Anna"]
CONTEXTS = ["morning", "afternoon"]
ITEMS = ["politics", "sports", "music", "food", "finance", "health", "camping"]
N_STEPS = 3000
DRIFT_POINT = 2000
ROLLING_WINDOW = 50
SEED = 42


def get_preferred_item(user, context, step):
    """Oracle: returns the item the user truly prefers at this step."""
    if step <= DRIFT_POINT:
        prefs = {
            ("Tom", "morning"): "politics",
            ("Tom", "afternoon"): "music",
            ("Anna", "morning"): "sports",
            ("Anna", "afternoon"): "finance",
        }
    else:
        # ALL four preferences change after drift
        prefs = {
            ("Tom", "morning"): "food",       # was politics
            ("Tom", "afternoon"): "health",    # was music
            ("Anna", "morning"): "camping",    # was sports
            ("Anna", "afternoon"): "politics", # was finance
        }
    return prefs[(user, context)]


def oracle_reward(user, item, context, step):
    """Binary reward: 1 if item matches current preference, 0 otherwise."""
    return int(item == get_preferred_item(user, context, step))


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────

class ContextualFunkMF(reco.FunkMF):
    """Matrix factorisation model that treats (user, time_of_day) as a
    composite key, giving it context-dependent recommendations."""

    def learn_one(self, user, item, y, x):
        return super().learn_one(f"{user}@{x['time_of_day']}", item, y, x)

    def rank(self, user, items, context):
        return super().rank(f"{user}@{context['time_of_day']}", items, context)


# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────

# Two separate RNGs so the user/context sequence is identical
# to the passive script regardless of action decisions.
env_rng = random.Random(SEED)        # draws users and contexts
action_rng = random.Random(SEED + 1) # handles epsilon-greedy decisions

model = ContextualFunkMF(seed=SEED, n_factors=5, optimizer=optim.SGD(lr=0.1))
EPSILON = 0.15  # constant exploration rate

records = []

for step in range(1, N_STEPS + 1):

    # --- environment draws a random user and context ---
    user = env_rng.choice(USERS)
    ctx_name = env_rng.choice(CONTEXTS)
    ctx = {"time_of_day": ctx_name}

    # --- the MODEL selects which item to show (epsilon-greedy) ---
    if action_rng.random() < EPSILON:
        selected_item = action_rng.choice(ITEMS)
    else:
        selected_item = model.rank(user, items=ITEMS, context=ctx)[0]

    # --- reward depends on the model's own choice ---
    reward = oracle_reward(user, selected_item, ctx_name, step)

    # --- learn from the consequence of its action ---
    model.learn_one(user, selected_item, y=reward, x=ctx)

    # --- evaluation: would the model's greedy pick be correct? ---
    greedy_pick = model.rank(user, items=ITEMS, context=ctx)[0]
    correct = int(greedy_pick == get_preferred_item(user, ctx_name, step))

    records.append({"step": step, "correct": correct})


# ──────────────────────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────────────────────

df = pd.DataFrame(records)
df["rolling_accuracy"] = (
    df["correct"]
    .rolling(window=ROLLING_WINDOW, min_periods=1)
    .mean()
)

os.makedirs("results", exist_ok=True)
df.to_csv("results/active_results.csv", index=False)

print(f"Active evaluation complete. {len(df)} steps recorded.")
print(f"Final rolling accuracy: {df['rolling_accuracy'].iloc[-1]:.3f}")
