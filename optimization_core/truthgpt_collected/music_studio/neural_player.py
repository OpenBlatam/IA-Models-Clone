import os, sys, subprocess
from pathlib import Path
def play():
    p = Path("../../../exports/neural_melody.wav").resolve()
    if p.exists():
        if sys.platform == "win32": os.startfile(p)
        else: subprocess.run(["open" if sys.platform=="darwin" else "xdg-open", str(p)])
if __name__ == "__main__": play()