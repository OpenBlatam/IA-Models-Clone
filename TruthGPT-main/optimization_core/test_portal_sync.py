import os
from interface.preferences import load_user_prefs, populate_env_from_prefs, save_user_prefs
from agents.razonamiento_planificacion.config import settings

# 1. Start with known state
prefs = load_user_prefs()
prefs["mcts_optimized"] = True
save_user_prefs(prefs)

# 2. Trigger population
populate_env_from_prefs(prefs)

print(f"TRUTHGPT_USE_MCTS_REASONING env: {os.environ.get('TRUTHGPT_USE_MCTS_REASONING')}")
print(f"settings.USE_MCTS_REASONING: {settings.USE_MCTS_REASONING}")

# 3. Disable it
prefs["mcts_optimized"] = False
save_user_prefs(prefs)
populate_env_from_prefs(prefs)

print(f"TRUTHGPT_USE_MCTS_REASONING env (after): {os.environ.get('TRUTHGPT_USE_MCTS_REASONING')}")
print(f"settings.USE_MCTS_REASONING (after): {settings.USE_MCTS_REASONING}")
