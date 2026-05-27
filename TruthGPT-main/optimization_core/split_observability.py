import os
import shutil

base_dir = "c:/blatam-academy/TruthGPT-main/optimization_core/agents/observability"
init_file = os.path.join(base_dir, "__init__.py")

with open(init_file, 'r', encoding='utf-8') as f:
    content = f.read()

# We need to extract the `Span` class into `models.py`
# And `Tracer` into `tracer.py`

imports = """import logging
import time
import uuid
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, computed_field

logger = logging.getLogger(__name__)
"""

# Extract Span class
span_start = content.find("class Span(BaseModel):")
tracer_start = content.find("class Tracer:")

if span_start == -1 or tracer_start == -1:
    print("Could not find classes")
    exit(1)

span_code = content[span_start:tracer_start].strip()
tracer_code = content[tracer_start:].strip()

# tracer code ends at `# Singleton tracer instance`
singleton_idx = tracer_code.find("# Singleton tracer instance")
if singleton_idx != -1:
    tracer_class_code = tracer_code[:singleton_idx].strip()
else:
    tracer_class_code = tracer_code

# Write models.py
with open(os.path.join(base_dir, "models.py"), 'w', encoding='utf-8') as f:
    f.write(imports + "\n\n" + span_code + "\n")

# Write tracer.py
with open(os.path.join(base_dir, "tracer.py"), 'w', encoding='utf-8') as f:
    f.write(imports + "\nfrom .models import Span\n\n" + tracer_class_code + "\n")

# Write new __init__.py
new_init = """\"\"\"
OpenClaw -- Agent Observability & Tracing.
\"\"\"
from .tracer import Tracer

# Singleton tracer instance for the entire application
global_tracer = Tracer()
"""
shutil.move(init_file, init_file + ".bak")
with open(init_file, 'w', encoding='utf-8') as f:
    f.write(new_init)

print("Observability refactoring completed.")
