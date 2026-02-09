from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import subprocess
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import structlog
import json
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Optional
import logging
"""
Git Manager for Key Messages ML Pipeline
Provides programmatic Git operations and integration
Updated with modern Python best practices and PyTorch ecosystem standards
"""


logger = structlog.get_logger(__name__)

@dataclass
class GitConfig:
    """Git configuration settings with modern defaults."""
    repo_path: str = "."
    user_name: str = "ML Pipeline"
    user_email: str = "ml-pipeline@example.com"
    auto_commit: bool = True
    auto_push: bool = False
    commit_message_template: str = "Auto-commit: {change_type} - {description}"
    branch: str = "main"
    remote: str = "origin"
    timeout: int = 30  # Command timeout in seconds
    max_retries: int = 3
    use_git_lfs: bool = True  # Git LFS for large files
    
    def __post_init__(self) -> Any:
        """Validate Git configuration."""
        if not self.user_name or not self.user_email:
            raise ValueError("user_name and user_email must be provided")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

@dataclass
class GitCommit:
    """Represents a Git commit with enhanced metadata."""
    hash: str
    author: str
    date: str
    message: str
    files_changed: List[str]
    insertions: int = 0
    deletions: int = 0
    merge: bool = False
    parents: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> Any:
        if not self.hash:
            raise ValueError("Commit hash is required")

@dataclass
class GitBranch:
    """Represents a Git branch with tracking information."""
    name: str
    is_current: bool
    last_commit: Optional[GitCommit] = None
    tracking_branch: Optional[str] = None
    ahead: int = 0
    behind: int = 0
    
    def __post_init__(self) -> Any:
        if not self.name:
            raise ValueError("Branch name is required")

@dataclass
class GitTag:
    """Represents a Git tag with enhanced metadata."""
    name: str
    commit_hash: str
    message: str
    date: str
    tag_type: str = "lightweight"  # lightweight, annotated
    author: Optional[str] = None
    
    def __post_init__(self) -> Any:
        if not self.name or not self.commit_hash:
            raise ValueError("Tag name and commit hash are required")

class GitManager:
    """Modern Git manager with enhanced features and error handling."""
    
    def __init__(self, config: GitConfig):
        
    """__init__ function."""
self.config = config
        self.repo_path = Path(config.repo_path).resolve()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Ensure repository path exists
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("GitManager initialized", 
                   repo_path=str(self.repo_path),
                   user_name=config.user_name,
                   user_email=config.user_email)
    
    def __enter__(self) -> Any:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        self._executor.shutdown(wait=True)
    
    async def _run_git_command_async(self, command: List[str], cwd: Optional[str] = None) -> str:
        """Run a Git command asynchronously."""
        try:
            cwd = cwd or str(self.repo_path)
            process = await asyncio.create_subprocess_exec(
                "git", *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.config.timeout
            )
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, ["git"] + command, stdout, stderr
                )
            
            return stdout.decode().strip()
            
        except asyncio.TimeoutError:
            logger.error("Git command timed out", command=command, timeout=self.config.timeout)
            raise
        except Exception as e:
            logger.error("Git command failed", command=command, error=str(e))
            raise
    
    def _run_git_command(self, command: List[str], cwd: Optional[str] = None, retries: int = None) -> str:
        """Run a Git command with retry logic."""
        if retries is None:
            retries = self.config.max_retries
        
        for attempt in range(retries + 1):
            try:
                cwd = cwd or str(self.repo_path)
                result = subprocess.run(
                    ["git"] + command,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout,
                    check=True
                )
                return result.stdout.strip()
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Git command timed out (attempt {attempt + 1}/{retries + 1})", 
                             command=command)
                if attempt == retries:
                    raise
                time.sleep(1)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Git command failed (attempt {attempt + 1}/{retries + 1})", 
                           command=command, error=e.stderr, return_code=e.returncode)
                if attempt == retries:
                    raise
                time.sleep(1)
    
    @contextmanager
    def _git_config_context(self, **kwargs) -> Any:
        """Context manager for temporary Git configuration."""
        original_config = {}
        
        try:
            # Store original config
            for key, value in kwargs.items():
                try:
                    original_config[key] = self._run_git_command(["config", f"user.{key}"])
                except subprocess.CalledProcessError:
                    original_config[key] = None
            
            # Set new config
            for key, value in kwargs.items():
                self._run_git_command(["config", f"user.{key}", str(value)])
            
            yield
            
        finally:
            # Restore original config
            for key, value in original_config.items():
                if value is not None:
                    self._run_git_command(["config", f"user.{key}", value])
                else:
                    try:
                        self._run_git_command(["config", "--unset", f"user.{key}"])
                    except subprocess.CalledProcessError:
                        pass  # Config didn't exist
    
    def is_repo(self) -> bool:
        """Check if the directory is a Git repository."""
        try:
            self._run_git_command(["rev-parse", "--git-dir"])
            return True
        except subprocess.CalledProcessError:
            return False
    
    def init_repo(self, bare: bool = False, initial_branch: str = None) -> bool:
        """Initialize a new Git repository with modern options."""
        try:
            if not self.is_repo():
                init_command = ["init"]
                if bare:
                    init_command.append("--bare")
                if initial_branch:
                    init_command.extend(["--initial-branch", initial_branch])
                
                self._run_git_command(init_command)
                
                # Set user configuration
                self._run_git_command(["config", "user.name", self.config.user_name])
                self._run_git_command(["config", "user.email", self.config.user_email])
                
                # Initialize Git LFS if enabled
                if self.config.use_git_lfs:
                    try:
                        self._run_git_command(["lfs", "install"])
                        logger.info("Git LFS initialized")
                    except subprocess.CalledProcessError:
                        logger.warning("Git LFS not available, continuing without LFS")
                
                # Create initial commit if there are files
                if self._has_staged_changes():
                    self._run_git_command(["commit", "-m", "Initial commit"])
                
                logger.info("Git repository initialized", 
                           repo_path=str(self.repo_path),
                           bare=bare,
                           initial_branch=initial_branch)
                return True
            else:
                logger.info("Git repository already exists", repo_path=str(self.repo_path))
                return False
                
        except Exception as e:
            logger.error("Failed to initialize Git repository", error=str(e))
            return False
    
    def _has_staged_changes(self) -> bool:
        """Check if there are staged changes."""
        try:
            result = self._run_git_command(["diff", "--cached", "--name-only"])
            return bool(result.strip())
        except subprocess.CalledProcessError:
            return False
    
    def _has_unstaged_changes(self) -> bool:
        """Check if there are unstaged changes."""
        try:
            result = self._run_git_command(["diff", "--name-only"])
            return bool(result.strip())
        except subprocess.CalledProcessError:
            return False
    
    def status(self) -> Dict[str, Any]:
        """Get comprehensive Git repository status."""
        try:
            # Get current branch
            current_branch = self._run_git_command(["branch", "--show-current"])
            
            # Get last commit details
            last_commit_hash = self._run_git_command(["rev-parse", "HEAD"])
            last_commit_message = self._run_git_command(["log", "-1", "--pretty=format:%s"])
            last_commit_author = self._run_git_command(["log", "-1", "--pretty=format:%an"])
            last_commit_date = self._run_git_command(["log", "-1", "--pretty=format:%ai"])
            last_commit_email = self._run_git_command(["log", "-1", "--pretty=format:%ae"])
            
            # Get staged and unstaged files
            staged_files = []
            unstaged_files = []
            
            if self._has_staged_changes():
                staged_files = self._run_git_command(["diff", "--cached", "--name-only"]).split("\n")
            
            if self._has_unstaged_changes():
                unstaged_files = self._run_git_command(["diff", "--name-only"]).split("\n")
            
            # Get untracked files
            untracked_files = []
            try:
                untracked_result = self._run_git_command(["ls-files", "--others", "--exclude-standard"])
                if untracked_result.strip():
                    untracked_files = untracked_result.split("\n")
            except subprocess.CalledProcessError:
                pass
            
            # Get repository info
            repo_info = {
                "current_branch": current_branch,
                "last_commit": {
                    "hash": last_commit_hash,
                    "message": last_commit_message,
                    "author": last_commit_author,
                    "email": last_commit_email,
                    "date": last_commit_date
                },
                "staged_files": staged_files,
                "unstaged_files": unstaged_files,
                "untracked_files": untracked_files,
                "has_changes": bool(staged_files or unstaged_files or untracked_files),
                "total_changes": len(staged_files) + len(unstaged_files) + len(untracked_files)
            }
            
            return repo_info
            
        except Exception as e:
            logger.error("Failed to get Git status", error=str(e))
            return {}
    
    def stage_file(self, file_path: str, force: bool = False) -> bool:
        """Stage a specific file with force option."""
        try:
            command = ["add"]
            if force:
                command.append("--force")
            command.append(file_path)
            
            self._run_git_command(command)
            logger.info("File staged", file_path=file_path, force=force)
            return True
            
        except Exception as e:
            logger.error("Failed to stage file", file_path=file_path, error=str(e))
            return False
    
    def stage_all(self, include_untracked: bool = True) -> bool:
        """Stage all changes with option to include untracked files."""
        try:
            command = ["add"]
            if include_untracked:
                command.append("--all")
            else:
                command.append(".")
            
            self._run_git_command(command)
            logger.info("All changes staged", include_untracked=include_untracked)
            return True
            
        except Exception as e:
            logger.error("Failed to stage all changes", error=str(e))
            return False
    
    def unstage_file(self, file_path: str) -> bool:
        """Unstage a specific file."""
        try:
            self._run_git_command(["reset", "HEAD", file_path])
            logger.info("File unstaged", file_path=file_path)
            return True
            
        except Exception as e:
            logger.error("Failed to unstage file", file_path=file_path, error=str(e))
            return False
    
    def commit(self, message: str, author: Optional[str] = None, 
               allow_empty: bool = False, sign: bool = False) -> Optional[str]:
        """Create a commit with enhanced options."""
        try:
            if not self._has_staged_changes() and not allow_empty:
                logger.warning("No staged changes to commit")
                return None
            
            # Prepare commit command
            command = ["commit", "-m", message]
            
            if allow_empty:
                command.append("--allow-empty")
            
            if sign:
                command.append("--gpg-sign")
            
            # Set author if provided
            env = None
            if author:
                env = os.environ.copy()
                env["GIT_AUTHOR_NAME"] = author
                env["GIT_COMMITTER_NAME"] = author
            
            # Create commit
            result = subprocess.run(
                command,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                env=env,
                timeout=self.config.timeout,
                check=True
            )
            
            # Extract commit hash
            commit_hash = self._run_git_command(["rev-parse", "HEAD"])
            
            logger.info("Commit created", 
                       hash=commit_hash,
                       message=message,
                       author=author or self.config.user_name,
                       sign=sign)
            
            return commit_hash
            
        except Exception as e:
            logger.error("Failed to create commit", message=message, error=str(e))
            return None
    
    def auto_commit(self, change_type: str, description: str, 
                   author: Optional[str] = None) -> Optional[str]:
        """Create an automatic commit using the configured template."""f"
        if not self.config.auto_commit:
            return None
        
        message = self.config.commit_message_template"
        
        return self.commit(message, author=author)
    
    def push(self, branch: Optional[str] = None, remote: Optional[str] = None,
             force: bool = False, tags: bool = False) -> bool:
        """Push changes to remote repository with enhanced options."""
        try:
            branch = branch or self.config.branch
            remote = remote or self.config.remote
            
            command = ["push", remote, branch]
            
            if force:
                command.append("--force-with-lease")  # Safer than --force
            
            if tags:
                command.append("--tags")
            
            self._run_git_command(command)
            
            logger.info("Changes pushed", remote=remote, branch=branch, force=force, tags=tags)
            return True
            
        except Exception as e:
            logger.error("Failed to push changes", error=str(e))
            return False
    
    def pull(self, branch: Optional[str] = None, remote: Optional[str] = None,
             rebase: bool = False, ff_only: bool = False) -> bool:
        """Pull changes from remote repository with enhanced options."""
        try:
            branch = branch or self.config.branch
            remote = remote or self.config.remote
            
            command = ["pull"]
            
            if rebase:
                command.append("--rebase")
            
            if ff_only:
                command.append("--ff-only")
            
            command.extend([remote, branch])
            
            self._run_git_command(command)
            
            logger.info("Changes pulled", remote=remote, branch=branch, rebase=rebase, ff_only=ff_only)
            return True
            
        except Exception as e:
            logger.error("Failed to pull changes", error=str(e))
            return False
    
    def create_branch(self, branch_name: str, checkout: bool = True, 
                     start_point: Optional[str] = None) -> bool:
        """Create a new branch with start point option."""
        try:
            command = ["checkout", "-b", branch_name]
            
            if start_point:
                command.append(start_point)
            
            self._run_git_command(command)
            
            if not checkout:
                # Switch back to original branch
                self._run_git_command(["checkout", self.config.branch])
            
            logger.info("Branch created", branch_name=branch_name, checkout=checkout, start_point=start_point)
            return True
            
        except Exception as e:
            logger.error("Failed to create branch", branch_name=branch_name, error=str(e))
            return False
    
    def checkout_branch(self, branch_name: str, create: bool = False) -> bool:
        """Checkout a branch with create option."""
        try:
            command = ["checkout"]
            
            if create:
                command.append("-b")
            
            command.append(branch_name)
            
            self._run_git_command(command)
            logger.info("Branch checked out", branch_name=branch_name, create=create)
            return True
            
        except Exception as e:
            logger.error("Failed to checkout branch", branch_name=branch_name, error=str(e))
            return False
    
    def list_branches(self, all_branches: bool = True, remote: bool = False) -> List[GitBranch]:
        """List branches with enhanced options."""
        try:
            command = ["branch"]
            
            if all_branches:
                command.append("-a")
            elif remote:
                command.append("-r")
            
            command.extend(["--format=%(refname:short)|%(upstream:short)|%(upstream:track)"])
            
            result = self._run_git_command(command)
            branches = []
            
            for line in result.split("\n"):
                if line.strip():
                    parts = line.split("|")
                    name = parts[0].strip()
                    
                    # Remove remote prefix if present
                    if name.startswith("remotes/"):
                        name = name[8:]
                    
                    # Parse tracking information
                    tracking_branch = parts[1].strip() if len(parts) > 1 and parts[1] else None
                    tracking_info = parts[2].strip() if len(parts) > 2 and parts[2] else ""
                    
                    # Parse ahead/behind
                    ahead = 0
                    behind = 0
                    if tracking_info:
                        if "ahead" in tracking_info:
                            ahead = int(tracking_info.split()[0])
                        if "behind" in tracking_info:
                            behind = int(tracking_info.split()[0])
                    
                    # Check if current branch
                    is_current = name == self._run_git_command(["branch", "--show-current"])
                    
                    branches.append(GitBranch(
                        name=name,
                        is_current=is_current,
                        tracking_branch=tracking_branch,
                        ahead=ahead,
                        behind=behind
                    ))
            
            return branches
            
        except Exception as e:
            logger.error("Failed to list branches", error=str(e))
            return []
    
    def create_tag(self, tag_name: str, message: str = "", 
                  commit: Optional[str] = None, force: bool = False) -> bool:
        """Create a tag with enhanced options."""
        try:
            command = ["tag"]
            
            if force:
                command.append("--force")
            
            if message:
                command.extend(["-a", tag_name, "-m", message])
            else:
                command.append(tag_name)
            
            if commit:
                command.append(commit)
            
            self._run_git_command(command)
            
            logger.info("Tag created", tag_name=tag_name, message=message, commit=commit, force=force)
            return True
            
        except Exception as e:
            logger.error("Failed to create tag", tag_name=tag_name, error=str(e))
            return False
    
    def list_tags(self, pattern: Optional[str] = None) -> List[GitTag]:
        """List tags with pattern filtering."""
        try:
            command = ["tag", "-l"]
            
            if pattern:
                command.append(pattern)
            
            command.extend(["--format=%(refname:strip=2)|%(objectname)|%(contents:subject)|%(creatordate:iso)|%(taggername)"])
            
            result = self._run_git_command(command)
            tags = []
            
            for line in result.split("\n"):
                if line.strip():
                    parts = line.split("|")
                    if len(parts) >= 4:
                        tag_type = "annotated" if len(parts) > 4 and parts[4] else "lightweight"
                        
                        tags.append(GitTag(
                            name=parts[0],
                            commit_hash=parts[1],
                            message=parts[2],
                            date=parts[3],
                            tag_type=tag_type,
                            author=parts[4] if len(parts) > 4 else None
                        ))
            
            return tags
            
        except Exception as e:
            logger.error("Failed to list tags", error=str(e))
            return []
    
    def get_commit_history(self, limit: int = 10, since: Optional[str] = None,
                          until: Optional[str] = None, author: Optional[str] = None) -> List[GitCommit]:
        """Get commit history with enhanced filtering."""
        try:
            format_str = "%H|%an|%ai|%s|%P|%n"
            command = ["log", f"-{limit}", f"--format={format_str}", "--name-only"]
            
            if since:
                command.append(f"--since={since}")
            
            if until:
                command.append(f"--until={until}")
            
            if author:
                command.append(f"--author={author}")
            
            result = self._run_git_command(command)
            
            commits = []
            lines = result.split("\n")
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                if line and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        # Get files changed
                        files_changed = []
                        j = i + 1
                        while j < len(lines) and lines[j].strip() and not "|" in lines[j]:
                            files_changed.append(lines[j].strip())
                            j += 1
                        
                        # Parse parents
                        parents = parts[4].split() if parts[4] else []
                        
                        commits.append(GitCommit(
                            hash=parts[0],
                            author=parts[1],
                            date=parts[2],
                            message=parts[3],
                            files_changed=files_changed,
                            parents=parents,
                            merge=len(parents) > 1
                        ))
                        
                        i = j
                    else:
                        i += 1
                else:
                    i += 1
            
            return commits
            
        except Exception as e:
            logger.error("Failed to get commit history", error=str(e))
            return []
    
    def get_file_history(self, file_path: str, limit: int = 10) -> List[GitCommit]:
        """Get history of a specific file."""
        try:
            format_str = "%H|%an|%ai|%s"
            result = self._run_git_command([
                "log", 
                f"-{limit}", 
                f"--format={format_str}",
                "--follow",
                file_path
            ])
            
            commits = []
            for line in result.split("\n"):
                if line.strip() and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        commits.append(GitCommit(
                            hash=parts[0],
                            author=parts[1],
                            date=parts[2],
                            message=parts[3],
                            files_changed=[file_path]
                        ))
            
            return commits
            
        except Exception as e:
            logger.error("Failed to get file history", file_path=file_path, error=str(e))
            return []
    
    def diff_file(self, file_path: str, commit1: str = "HEAD", commit2: str = "HEAD~1",
                  stat: bool = False) -> str:
        """Get diff between two commits for a specific file."""
        try:
            command = ["diff", commit1, commit2, "--", file_path]
            
            if stat:
                command.append("--stat")
            
            result = self._run_git_command(command)
            return result
            
        except Exception as e:
            logger.error("Failed to get file diff", 
                        file_path=file_path,
                        commit1=commit1,
                        commit2=commit2,
                        error=str(e))
            return ""
    
    def diff_all(self, commit1: str = "HEAD", commit2: str = "HEAD~1",
                 stat: bool = False, name_only: bool = False) -> str:
        """Get diff between two commits for all files."""
        try:
            command = ["diff", commit1, commit2]
            
            if stat:
                command.append("--stat")
            elif name_only:
                command.append("--name-only")
            
            result = self._run_git_command(command)
            return result
            
        except Exception as e:
            logger.error("Failed to get diff", 
                        commit1=commit1,
                        commit2=commit2,
                        error=str(e))
            return ""
    
    def reset_hard(self, commit: str = "HEAD") -> bool:
        """Reset repository to a specific commit (hard reset)."""
        try:
            self._run_git_command(["reset", "--hard", commit])
            logger.info("Repository reset", commit=commit)
            return True
            
        except Exception as e:
            logger.error("Failed to reset repository", commit=commit, error=str(e))
            return False
    
    def reset_soft(self, commit: str = "HEAD") -> bool:
        """Reset repository to a specific commit (soft reset)."""
        try:
            self._run_git_command(["reset", "--soft", commit])
            logger.info("Repository soft reset", commit=commit)
            return True
            
        except Exception as e:
            logger.error("Failed to soft reset repository", commit=commit, error=str(e))
            return False
    
    def stash(self, message: str = "", include_untracked: bool = False) -> bool:
        """Stash current changes with enhanced options."""
        try:
            command = ["stash", "push"]
            
            if message:
                command.extend(["-m", message])
            
            if include_untracked:
                command.append("-u")
            
            self._run_git_command(command)
            
            logger.info("Changes stashed", message=message, include_untracked=include_untracked)
            return True
            
        except Exception as e:
            logger.error("Failed to stash changes", error=str(e))
            return False
    
    def stash_pop(self, stash_id: Optional[str] = None) -> bool:
        """Pop the latest stash or specific stash."""
        try:
            command = ["stash", "pop"]
            
            if stash_id:
                command.append(stash_id)
            
            self._run_git_command(command)
            logger.info("Stash popped", stash_id=stash_id)
            return True
            
        except Exception as e:
            logger.error("Failed to pop stash", error=str(e))
            return False
    
    def list_stashes(self) -> List[Dict[str, str]]:
        """List all stashes with enhanced information."""
        try:
            result = self._run_git_command([
                "stash", "list", 
                "--format=format:%gd|%gs|%ai|%an"
            ])
            stashes = []
            
            for line in result.split("\n"):
                if line.strip() and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        stashes.append({
                            "id": parts[0],
                            "message": parts[1],
                            "date": parts[2],
                            "author": parts[3]
                        })
            
            return stashes
            
        except Exception as e:
            logger.error("Failed to list stashes", error=str(e))
            return []
    
    def get_repo_info(self) -> Dict[str, Any]:
        """Get comprehensive repository information."""
        try:
            status = self.status()
            branches = self.list_branches()
            tags = self.list_tags()
            commits = self.get_commit_history(5)
            
            # Get repository size
            repo_size = 0
            try:
                git_dir = self._run_git_command(["rev-parse", "--git-dir"])
                git_path = Path(self.repo_path) / git_dir
                repo_size = sum(f.stat().st_size for f in git_path.rglob('*') if f.is_file())
            except Exception:
                pass
            
            return {
                "repo_path": str(self.repo_path),
                "is_repo": self.is_repo(),
                "repo_size_bytes": repo_size,
                "config": {
                    "user_name": self.config.user_name,
                    "user_email": self.config.user_email,
                    "branch": self.config.branch,
                    "remote": self.config.remote
                },
                "status": status,
                "branches": [{"name": b.name, "is_current": b.is_current, 
                            "tracking": b.tracking_branch, "ahead": b.ahead, "behind": b.behind} 
                           for b in branches],
                "tags": [{"name": t.name, "message": t.message, "type": t.tag_type} for t in tags],
                "recent_commits": [{"hash": c.hash, "message": c.message, 
                                  "author": c.author, "merge": c.merge} for c in commits]
            }
            
        except Exception as e:
            logger.error("Failed to get repository info", error=str(e))
            return {}
    
    def cleanup(self, prune: bool = True, aggressive: bool = False) -> bool:
        """Clean up repository with garbage collection."""
        try:
            if prune:
                self._run_git_command(["gc", "--prune=now"])
            
            if aggressive:
                self._run_git_command(["gc", "--aggressive"])
            
            logger.info("Repository cleanup completed", prune=prune, aggressive=aggressive)
            return True
            
        except Exception as e:
            logger.error("Failed to cleanup repository", error=str(e))
            return False 