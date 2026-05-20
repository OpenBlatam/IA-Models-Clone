import os
import re

directory = r"C:\blatam-academy\agents\backend\onyx\server\features\lovable\web_gen_system"
replacement_map = {
    # Original -> Replacement
    r"from \.\.agents\.base import BaseAgent": r"from agents.backend.onyx.server.features.lovable.web_gen_system.agents.base import BaseAgent",
    r"from \.\.schemas import AgentContext": r"from agents.backend.onyx.server.features.lovable.web_gen_system.schemas import AgentContext",
    r"from \.\.core\.event_bus import EventBus": r"from agents.backend.onyx.server.features.lovable.web_gen_system.core.event_bus import EventBus",
    r"from \.\.core\.memory import SharedMemory": r"from agents.backend.onyx.server.features.lovable.web_gen_system.core.memory import SharedMemory",
    r"from \.base import BaseAgent": r"from agents.backend.onyx.server.features.lovable.web_gen_system.agents.base import BaseAgent",
    r"from \.\.frameworks\.nextjs import NextJSGenerator": r"from agents.backend.onyx.server.features.lovable.web_gen_system.frameworks.nextjs import NextJSGenerator",
    r"from \.\.frameworks\.expo import ExpoGenerator": r"from agents.backend.onyx.server.features.lovable.web_gen_system.frameworks.expo import ExpoGenerator",
    r"from \.\.accessibility import WebAccessibilityEnhancer": r"from agents.backend.onyx.server.features.lovable.web_gen_system.accessibility import WebAccessibilityEnhancer",
    r"from \.\.instruction import InstructionParser": r"from agents.backend.onyx.server.features.lovable.web_gen_system.instruction import InstructionParser",
    r"from \.\.context import RepositoryContext": r"from agents.backend.onyx.server.features.lovable.web_gen_system.context import RepositoryContext",
    r"from \.accessibility import": r"from agents.backend.onyx.server.features.lovable.web_gen_system.accessibility import",
    r"from \.seo import": r"from agents.backend.onyx.server.features.lovable.web_gen_system.seo import",
    r"from \.generator import": r"from agents.backend.onyx.server.features.lovable.web_gen_system.generator import",
    r"from \.pipeline import": r"from agents.backend.onyx.server.features.lovable.web_gen_system.pipeline import",
}

def mass_refactor(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py") and file != "mass_refactor.py":
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = content
                for pattern, replacement in replacement_map.items():
                    new_content = re.sub(pattern, replacement, new_content)
                
                if new_content != content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated {path}")

if __name__ == "__main__":
    mass_refactor(directory)
