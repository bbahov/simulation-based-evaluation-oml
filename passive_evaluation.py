# passive_evaluation.py
#
# Passive Evaluation Environment
# ===============================
# The model observes randomly drawn (user, item, reward) triples.
# It learns from this stream but never selects which items to show.
# The data it sees is entirely independent of its predictions.
#
# Run locally:   python passive_evaluation.py
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
# MODEL  (same architecture as active script)
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

# Same env_rng seed as the active script => identical user/context sequence.
# item_rng replaces action_rng: it draws random items instead of
# making epsilon-greedy decisions.
env_rng = random.Random(SEED)        # draws users and contexts
item_rng = random.Random(SEED + 1)   # draws random items

model = ContextualFunkMF(seed=SEED, n_factors=5, optimizer=optim.SGD(lr=0.1))

records = []

for step in range(1, N_STEPS + 1):

    # --- environment draws a random user and context ---
    # (identical sequence to the active script)
    user = env_rng.choice(USERS)
    ctx_name = env_rng.choice(CONTEXTS)
    ctx = {"time_of_day": ctx_name}

    # --- item is drawn at random: model has NO influence ---
    observed_item = item_rng.choice(ITEMS)

    # --- reward for the randomly drawn item ---
    reward = oracle_reward(user, observed_item, ctx_name, step)

    # --- learn from someone else's exploration ---
    model.learn_one(user, observed_item, y=reward, x=ctx)

    # --- evaluation: would the model's greedy pick be correct? ---
    # (same metric as active script — measures learned knowledge)
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
df.to_csv("results/passive_results.csv", index=False)

print(f"Passive evaluation complete. {len(df)} steps recorded.")
print(f"Final rolling accuracy: {df['rolling_accuracy'].iloc[-1]:.3f}")
