import os
import zipfile
import shutil

def create_enterprise_package():
    output_zip = r"C:\blatam-academy\Enterprise_Review_Package.zip"
    script_dir = r"C:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run-polyglot\scripts"
    
    files_to_zip = [
        (r"C:\blatam-academy\agents\frontend\shopping-engine\src\hooks\useTaskPolling.ts", "shopping-engine/src/hooks/useTaskPolling.ts"),
        (r"C:\blatam-academy\agents\backend\onyx\server\features\gamma_app\services\analytics_service.py", "gamma_app/services/analytics_service.py"),
        (r"C:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run-polyglot\scripts\TruthGPT-main\optimization_core\agents\embodied_rl\rl_agent.py", "TruthGPT-main/optimization_core/agents/embodied_rl/rl_agent.py"),
        (r"C:\blatam-academy\agents\backend\onyx\server\features\github_autonomous_agent_ai\frontend\app\kanban\page.tsx", "github_autonomous_agent_ai/frontend/app/kanban/page.tsx"),
        (r"C:\blatam-academy\agents\backend\onyx\server\features\github_autonomous_agent_ai\unified_backend\github_autonomous_agent\main.py", "github_autonomous_agent_ai/unified_backend/github_autonomous_agent/main.py")
    ]
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for absolute_path, arcname in files_to_zip:
            if os.path.exists(absolute_path):
                zf.write(absolute_path, arcname)
                print(f"Added: {arcname}")
            else:
                print(f"Warning: File not found {absolute_path}")
                
    print(f"\n✅ Enterprise Package created successfully at: {output_zip}")

if __name__ == '__main__':
    create_enterprise_package()
