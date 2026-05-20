
import os
import json
import re

FEATURES_DIR = r"C:\blatam-academy\agents\backend\onyx\server\features"
OUTPUT_README = os.path.join(FEATURES_DIR, "README.md")
REPORT_FILE = "feature_analysis_report.json"

IGNORED_DIRS = {
    ".agent", ".git", ".github", ".husky", "node_modules", "__pycache__", 
    "assets", "config", "docs", "frontend", "hooks", "lib", "public", 
    "scripts", "styles", "types", "utils", "bin", "obj", "include", "share",
    "nginx", "prisma", "actions", "app", "components", "pages"
}

class FeatureAnalyzer:
    def __init__(self, directory):
        self.directory = directory
        self.features = []

    def analyze(self):
        try:
            items = os.listdir(self.directory)
        except FileNotFoundError:
            return

        for item in items:
            item_path = os.path.join(self.directory, item)
            
            if not os.path.isdir(item_path):
                continue
                
            if item in IGNORED_DIRS or item.startswith("."):
                continue

            feature_info = {
                "name": item,
                "path": item_path,
                "title": item.replace("_", " ").title(),
                "has_readme": False,
                "readme_content": "",
                "tech_stack": [],
                "connections": set(),
                "description": ""
            }

            # Check README
            readme_path = os.path.join(item_path, "README.md")
            if os.path.exists(readme_path):
                feature_info["has_readme"] = True
                try:
                    with open(readme_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        feature_info["readme_content"] = content[:200]
                except Exception:
                    pass

            # Deep verification of code
            for root, _, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Detect Tech Stack
                    if file.endswith(".py"):
                        if "Python" not in feature_info["tech_stack"]:
                            feature_info["tech_stack"].append("Python")
                        
                        # Extract description from docstring of main files
                        if file in ["main.py", "__init__.py", "app.py", "server.py"] and not feature_info["description"]:
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                                    if docstring_match:
                                        feature_info["description"] = docstring_match.group(1).strip()
                            except:
                                pass
                                
                        # Detect connections (imports)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                # Look for imports from other features
                                # Assumes features are installed or in python path as 'features.xyz' or similar
                                imports = re.findall(r'from\s+.*?features\.(\w+)', content)
                                imports += re.findall(r'import\s+.*?features\.(\w+)', content)
                                for imp in imports:
                                    if imp != item and imp in items:
                                        feature_info["connections"].add(imp)
                        except:
                            pass

                    elif file.endswith(".js") or file.endswith(".ts"):
                        if "Node.js" not in feature_info["tech_stack"]:
                            feature_info["tech_stack"].append("Node.js")
                    elif file == "Dockerfile":
                        if "Docker" not in feature_info["tech_stack"]:
                            feature_info["tech_stack"].append("Docker")

                # Limit depth
                if root.count(os.sep) - item_path.count(os.sep) > 2:
                    break

            feature_info["connections"] = list(feature_info["connections"])
            self.features.append(feature_info)

        return self.features

    def generate_main_readme(self):
        self.features.sort(key=lambda x: x["name"])
        
        content = "# 🚀 Server Features\n\n"
        content += "This directory contains the modular features and agents of the Onyx system.\n\n"
        
        # summary stats
        total = len(self.features)
        documented = sum(1 for f in self.features if f["has_readme"])
        content += f"**Total Features:** {total} | **Documented:** {documented}\n\n"

        content += "## 📦 Feature Index\n\n"
        content += "| Feature | Description | Tech Stack | Status |\n"
        content += "|---|---|---|---|\n"

        for feature in self.features:
            description = "No description available."
            # Prefer extracted docstring, fall back to README
            if feature["description"]:
                description = feature["description"]
            elif feature["readme_content"]:
                lines = feature["readme_content"].split('\n')
                for line in lines:
                    cleaned = line.strip().lstrip('#').strip()
                    if cleaned and cleaned != feature["title"]:
                        description = cleaned
                        break
            
            # Truncate for table
            short_desc = (description[:75] + '...') if len(description) > 75 else description
            
            tech = ", ".join(feature["tech_stack"]) if feature["tech_stack"] else "Unknown"
            status = "✅ Documented" if feature["has_readme"] else "⚠️ Missing README"
            
            link = f"[{feature['title']}](./{feature['name']}/README.md)"
            content += f"| {link} | {short_desc} | {tech} | {status} |\n"

        content += "\n## 🔗 Inter-Feature Connections\n\n"
        content += "The following graph shows automatically detected dependencies between features:\n\n"
        content += "```mermaid\n"
        content += "graph TD;\n"
        
        has_connections = False
        for feature in self.features:
            for conn in feature["connections"]:
                content += f"    {feature['name']} --> {conn};\n"
                has_connections = True
        
        if not has_connections:
            content += "    %% No explicit imports found between feature modules.\n"
            
        content += "```\n"
        
        with open(OUTPUT_README, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Generated README at {OUTPUT_README}")

    def save_report(self):
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.features, f, indent=2, default=str)
        print(f"Saved analysis report to {REPORT_FILE}")

if __name__ == "__main__":
    analyzer = FeatureAnalyzer(FEATURES_DIR)
    analyzer.analyze()
    analyzer.generate_main_readme()
    analyzer.save_report()
