import os
from pathlib import Path

# Path to loader.py
loader_path = Path(r"C:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run-polyglot\scripts\TruthGPT-main\optimization_core\modules\base\config_management\configs\loader.py")
print(f"Loader Path: {loader_path}")
print(f"Loader Parent: {loader_path.parent}")
print(f"llm_default.yaml: {loader_path.parent / 'llm_default.yaml'}")
print(f"Exists: {(loader_path.parent / 'llm_default.yaml').exists()}")

# Test relative path from C:\blatam-academy
rel_path = "modules/base/config_management/configs/llm_default.yaml"
print(f"Current CWD: {os.getcwd()}")
print(f"Rel path exists: {os.path.exists(rel_path)}")
