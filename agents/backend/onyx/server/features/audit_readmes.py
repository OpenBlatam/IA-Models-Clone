
import os

FEATURES_DIR = r"C:\blatam-academy\agents\backend\onyx\server\features"
BACK_LINK = "[← Back to Main README](../README.md)"

def audit_readmes():
    missing_readmes = []
    missing_links = []
    
    # helper for specific known infrastructure dirs to ignore
    ignored_dirs = {
        ".agent", ".git", ".github", ".husky", "node_modules", "__pycache__", 
        "assets", "config", "docs", "frontend", "hooks", "lib", "public", 
        "scripts", "styles", "types", "utils", "bin", "obj", "include", "share"
    }

    try:
        items = os.listdir(FEATURES_DIR)
    except FileNotFoundError:
        print(f"Directory not found: {FEATURES_DIR}")
        return

    for item in items:
        item_path = os.path.join(FEATURES_DIR, item)
        
        if not os.path.isdir(item_path):
            continue
            
        if item in ignored_dirs or item.startswith("."):
            continue
            
        readme_path = os.path.join(item_path, "README.md")
        
        if not os.path.exists(readme_path):
            missing_readmes.append(item)
        else:
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "../README.md" not in content:
                        missing_links.append(item)
            except Exception as e:
                print(f"Error reading {readme_path}: {e}")

    print("--- MISSING READMEs ---")
    for m in missing_readmes:
        print(m)
        
    print("\n--- MISSING BACK LINKS ---")
    for l in missing_links:
        print(l)

if __name__ == "__main__":
    audit_readmes()
