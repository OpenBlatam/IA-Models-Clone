from typing import List, Dict, Any
from ..agents.base import BaseAgent

class SecurityScannerAgent(BaseAgent):
    """
    Agent responsible for scanning generated code for security vulnerabilities.
    Inspired by SafeArena and SecureWebArena papers.
    """

    def __init__(self, name: str = "SecurityScanner"):
        super().__init__(name, "Security Engineer")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scans the generated code in the repository context for potential security issues.
        """
        try:
            self.log("Starting security scan...")
            
            repo_context = context.get("repository_context")
            if not repo_context:
                self.log("No repository context found. Skipping scan.", level="warning")
                return {"status": "skipped", "issues": []}

            files = repo_context.get_all_files()
            issues = []

            for file_path, content in files.items():
                file_issues = self._scan_file(file_path, content)
                if file_issues:
                    issues.extend(file_issues)

            if issues:
                self.log(f"Found {len(issues)} security issues.", level="warning")
                for issue in issues:
                    self.log(f"  - {issue}", level="warning")
                return {"status": "failed", "issues": issues}
            
            self.log("Security scan passed. No issues found.")
            return {"status": "passed", "issues": []}
            
        except Exception as e:
            self.log(f"Error in SecurityScannerAgent: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

    def _scan_file(self, file_path: str, content: str) -> List[str]:
        """
        Scans a single file for known vulnerability patterns.
        """
        issues = []
        
        # 1. Check for dangerouslySetInnerHTML (XSS risk)
        if "dangerouslySetInnerHTML" in content:
            issues.append(f"Potential XSS risk in {file_path}: Usage of 'dangerouslySetInnerHTML'.")

        # 2. Check for hardcoded secrets (Basic heuristic)
        if "API_KEY" in content or "SECRET_KEY" in content:
             # Ignore if it's just a type definition or environment variable access
            if "process.env" not in content and "import" not in content:
                 issues.append(f"Potential hardcoded secret in {file_path}: Found 'API_KEY' or 'SECRET_KEY'.")

        # 3. Check for eval() usage
        if "eval(" in content:
            issues.append(f"Dangerous code execution in {file_path}: Usage of 'eval()'.")

        # 4. Check for alert() usage (Quality/Annoyance, but sometimes flagged in security)
        if "alert(" in content:
            issues.append(f"Usage of 'alert()' in {file_path}. Prefer custom UI components.")

        return issues
