"""Script to verify MOEA project was generated correctly"""
import json
from pathlib import Path
import sys

PROJECT_DIR = Path("generated_projects/moea_optimization_system")

def check_directory(path: Path, name: str) -> bool:
    """Check if directory exists"""
    if path.exists() and path.is_dir():
        print(f"✅ {name}: {path}")
        return True
    else:
        print(f"❌ {name}: Missing - {path}")
        return False

def check_file(path: Path, name: str) -> bool:
    """Check if file exists"""
    if path.exists() and path.is_file():
        print(f"✅ {name}: {path}")
        return True
    else:
        print(f"❌ {name}: Missing - {path}")
        return False

def verify_project():
    """Verify MOEA project structure"""
    print("=" * 60)
    print("MOEA Project Verification")
    print("=" * 60)
    print()
    
    if not PROJECT_DIR.exists():
        print(f"❌ Project directory not found: {PROJECT_DIR}")
        print("   Run the generation script first!")
        return False
    
    print(f"📁 Project directory: {PROJECT_DIR.absolute()}")
    print()
    
    # Required directories
    required_dirs = [
        (PROJECT_DIR / "backend", "Backend directory"),
        (PROJECT_DIR / "backend" / "app", "Backend app directory"),
        (PROJECT_DIR / "backend" / "app" / "api", "Backend API directory"),
        (PROJECT_DIR / "backend" / "app" / "core", "Backend core directory"),
        (PROJECT_DIR / "frontend", "Frontend directory"),
        (PROJECT_DIR / "frontend" / "src", "Frontend src directory"),
        (PROJECT_DIR / "frontend" / "src" / "components", "Frontend components"),
        (PROJECT_DIR / "frontend" / "src" / "pages", "Frontend pages"),
    ]
    
    # Required files
    required_files = [
        (PROJECT_DIR / "README.md", "Project README"),
        (PROJECT_DIR / "backend" / "main.py", "Backend main.py"),
        (PROJECT_DIR / "backend" / "requirements.txt", "Backend requirements.txt"),
        (PROJECT_DIR / "frontend" / "package.json", "Frontend package.json"),
        (PROJECT_DIR / "frontend" / "vite.config.ts", "Frontend vite config"),
    ]
    
    print("Checking directories...")
    dirs_ok = all(check_directory(path, name) for path, name in required_dirs)
    print()
    
    print("Checking files...")
    files_ok = all(check_file(path, name) for path, name in required_files)
    print()
    
    # Check project_info.json
    project_info = PROJECT_DIR / "project_info.json"
    if project_info.exists():
        print("📄 Project info found:")
        try:
            with open(project_info, 'r', encoding='utf-8') as f:
                info = json.load(f)
                print(f"   Project Name: {info.get('project_name', 'N/A')}")
                print(f"   Author: {info.get('author', 'N/A')}")
                print(f"   Version: {info.get('version', 'N/A')}")
                print(f"   AI Type: {info.get('ai_type', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️  Error reading project info: {e}")
    else:
        print("⚠️  project_info.json not found")
    print()
    
    # Summary
    print("=" * 60)
    if dirs_ok and files_ok:
        print("✅ Project structure is valid!")
        print()
        print("Next steps:")
        print("1. Install backend dependencies:")
        print(f"   cd {PROJECT_DIR / 'backend'}")
        print("   pip install -r requirements.txt")
        print()
        print("2. Install frontend dependencies:")
        print(f"   cd {PROJECT_DIR / 'frontend'}")
        print("   npm install")
        print()
        print("3. Start development servers:")
        print("   Backend: uvicorn app.main:app --reload")
        print("   Frontend: npm run dev")
        return True
    else:
        print("❌ Project structure is incomplete!")
        print("   Some required files or directories are missing.")
        return False

if __name__ == "__main__":
    success = verify_project()
    sys.exit(0 if success else 1)

