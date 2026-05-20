from typing import Dict, List, Set
import re

class RepositoryContext:
    """
    Maintains the state of the generated project, allowing agents to understand
    file relationships and dependencies (CodePlan concept).
    """
    def __init__(self):
        self.files: Dict[str, str] = {}
        self.dependencies: Dict[str, Set[str]] = {}

    def add_file(self, path: str, content: str):
        """
        Adds or updates a file in the repository context.
        """
        self.files[path] = content
        self._analyze_dependencies(path, content)

    def get_file(self, path: str) -> str:
        """
        Retrieves the content of a file.
        """
        return self.files.get(path, "")

    def file_exists(self, path: str) -> bool:
        """
        Checks if a file exists in the repository.
        """
        return path in self.files

    def get_all_files(self) -> Dict[str, str]:
        """
        Returns all files in the repository.
        """
        return self.files

    def _analyze_dependencies(self, path: str, content: str):
        """
        Simple regex-based dependency analysis.
        """
        # Look for import statements (TypeScript/JavaScript)
        # import ... from '...'
        imports = set()
        matches = re.findall(r"from\s+['\"]([^'\"]+)['\"]", content)
        for match in matches:
            imports.add(match)
        
        self.dependencies[path] = imports

    def check_consistency(self) -> List[str]:
        """
        Checks for broken imports (simplified).
        """
        issues = []
        for file_path, imports in self.dependencies.items():
            for imp in imports:
                # Ignore external libraries (start with non-relative path)
                if not imp.startswith("."):
                    continue
                
                # Resolve relative path (very basic)
                # In a real system, we'd need a proper path resolver
                if imp.startswith("./"):
                    target = imp[2:]
                else:
                    target = imp
                
                # Check if target exists (fuzzy match for extensions)
                found = False
                for existing_file in self.files.keys():
                    if target in existing_file:
                        found = True
                        break
                
                if not found:
                    issues.append(f"File {file_path} imports missing module '{imp}'")
                    
        return issues
