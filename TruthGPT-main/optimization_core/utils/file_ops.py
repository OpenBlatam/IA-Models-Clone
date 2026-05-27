import re
import time
from pathlib import Path
from interface.core import console

def extract_filename_from_code(code_text: str, default_name: str) -> str:
    lines = code_text.strip().splitlines()
    if not lines:
        return default_name
    for line in lines[:3]:
        line = line.strip()
        match = re.match(r'^(?:#|//|/\*|File:|Filename:)\s*([a-zA-Z0-9_\-\.\/\\ ]+)(?:\s*\*\/)?$', line, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if '.' in extracted:
                parts = extracted.split('.')
                if len(parts[-1]) in (1, 2, 3, 4) and not parts[-1].isdigit():
                    return extracted
    return default_name

def save_code_blocks_to_directory(content: str, target_dir: Path, default_prefix: str = "output"):
    code_blocks = re.findall(r"```([a-zA-Z0-9+#_ -]*)\n(.*?)\n```", content, re.DOTALL)
    if not code_blocks:
        single_match = re.search(r"```(?:[a-zA-Z0-9+#_ -]*)\n(.*?)\n```", content, re.DOTALL)
        if single_match:
            code_blocks = [("", single_match.group(1))]
            
    lang_map = {
        "python": ".py", "py": ".py",
        "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts",
        "html": ".html", "htm": ".html",
        "css": ".css", "json": ".json",
        "rust": ".rs", "rs": ".rs", "go": ".go",
        "bash": ".sh", "sh": ".sh", "shell": ".sh",
        "powershell": ".ps1", "ps1": ".ps1",
    }
    
    saved_files = []
    for idx, (lang, code) in enumerate(code_blocks, 1):
        lang_clean = lang.strip().lower()
        ext = lang_map.get(lang_clean, ".py" if not lang_clean else f".{lang_clean}")
        default_filename = f"{default_prefix}_{idx}_{int(time.time())}{ext}"
        
        rel_filename = extract_filename_from_code(code, default_filename)
        dest_path = target_dir / rel_filename
        
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "w", encoding="utf-8") as code_f:
                code_f.write(code)
            saved_files.append(dest_path)
            console.print(f"  [green]● Saved code block to {dest_path.resolve()}[/green]")
            try:
                from interface.cc_style import cc_code_change
                cc_code_change("WRITE", str(dest_path.name), added=len(code.splitlines()))
            except Exception:
                pass
        except Exception as e:
            console.print(f"[red]Failed to save code to {dest_path}: {e}[/red]")
            
    return saved_files
