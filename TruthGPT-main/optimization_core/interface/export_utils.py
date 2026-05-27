"""
Export & Reporting Utilities for TruthGPT Interface.

Provides:
- export_mission_result(): Export mission content to MD/PDF/Word
- save_mission_output(): Auto-save mission output to reports directory
- extract_target_directory(): Parse a query string for filesystem paths
- Shared code-block extraction (eliminates duplication from original core.py)
"""

import re
import time
from pathlib import Path
from typing import Optional


# ─── Shared Constants ────────────────────────────────────────────────────

# Language extension mapping for code-block extraction (single source of truth)
LANG_EXT_MAP = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "html": ".html", "htm": ".html",
    "css": ".css",
    "json": ".json",
    "rust": ".rs", "rs": ".rs",
    "go": ".go",
    "bash": ".sh", "sh": ".sh", "shell": ".sh",
    "powershell": ".ps1", "ps1": ".ps1",
    "c": ".c", "cpp": ".cpp", "c++": ".cpp",
    "java": ".java",
}

# Project root for default report paths
_PROJECT_DIR = Path(__file__).resolve().parent.parent


# ─── Code Block Extraction (shared helper) ───────────────────────────────

def _extract_and_save_code_blocks(
    content: str,
    output_dir: Path,
    console,
    timestamp: Optional[str] = None,
) -> None:
    """Extract fenced code blocks from markdown content and save to output_dir."""
    code_blocks = re.findall(r"```([a-zA-Z0-9+#_ -]*)\n(.*?)\n```", content, re.DOTALL)
    if not code_blocks:
        return

    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    console.print(f"[cyan]📦 Extracting and writing {len(code_blocks)} code blocks to {output_dir}...[/cyan]")

    for idx, (lang, code) in enumerate(code_blocks, 1):
        lang_clean = lang.strip().lower()
        code_ext = LANG_EXT_MAP.get(lang_clean, ".py" if not lang_clean else f".{lang_clean}")
        code_filename = f"code_block_{idx}_{timestamp}{code_ext}"
        code_filepath = output_dir / code_filename
        with open(code_filepath, "w", encoding="utf-8") as code_f:
            code_f.write(code)
        console.print(
            f"  [green]● Saved code block {idx} ({lang_clean or 'python/unknown'}) to {code_filepath.name}[/green]"
        )


# ─── Path Extraction ─────────────────────────────────────────────────────

def extract_target_directory(query: Optional[str]) -> Optional[Path]:
    """Parse a user query string to find a filesystem directory path."""
    if not query:
        return None

    words = query.split()
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            candidate = " ".join(words[start : start + length]).strip("\"'")
            if not candidate:
                continue

            is_path_like = False
            if (
                re.match(r"^[a-zA-Z]:\\", candidate)
                or re.match(r"^[a-zA-Z]:/", candidate)
                or candidate.startswith("/")
                or candidate.startswith(".\\")
                or candidate.startswith("./")
            ):
                is_path_like = True
            elif "\\" in candidate or "/" in candidate:
                if not candidate.startswith("http"):
                    is_path_like = True

            if is_path_like:
                try:
                    path = Path(candidate)
                    if path.exists() and path.is_dir():
                        return path.resolve()
                    if not path.exists():
                        parent = path.parent
                        if parent and parent.exists():
                            return path.resolve()
                except Exception:
                    pass

    return None


# ─── Export Functions ─────────────────────────────────────────────────────

def export_mission_result(content: str, mission_name: str = "Mission_Result") -> None:
    """Export mission content to MD (PDF/Word stubs for future)."""
    from datetime import datetime
    from interface.input_handler import get_input
    from interface.core import console

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mission_name = mission_name.replace(" ", "_")

    console.print("\n[bold cyan]📤 Export & Reporting Engine[/bold cyan]")
    fmt = get_input("Export format", choices=["MD", "PDF", "Word"], default="MD").upper()
    filename = f"{mission_name}_{timestamp}"

    try:
        if fmt == "MD":
            path = Path(f"exports/{filename}.md")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            console.print(f"[bold green]✓ Exported to {path}[/bold green]")

            _extract_and_save_code_blocks(content, path.parent, console, timestamp)
    except Exception as e:
        console.print(f"[red]Export Error: {e}[/red]")


def save_mission_output(
    content: str, mission_name: str = "Mission", query: Optional[str] = None
) -> None:
    """Auto-save mission output to reports directory or a path found in the query."""
    from interface.core import console

    target_dir = extract_target_directory(query)
    if target_dir:
        report_dir = target_dir
    else:
        report_dir = _PROJECT_DIR / "reports"

    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{mission_name}_{timestamp}.md"
    filepath = report_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[bold green]✓ Output exported to {filepath}[/bold green]")

    _extract_and_save_code_blocks(content, report_dir, console, timestamp)
