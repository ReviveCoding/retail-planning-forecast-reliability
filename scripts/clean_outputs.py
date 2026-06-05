from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
for rel in ["outputs", "reports", ".pytest_cache"]:
    path = ROOT / rel
    if path.exists():
        shutil.rmtree(path)
for pycache in ROOT.rglob("__pycache__"):
    shutil.rmtree(pycache, ignore_errors=True)
for pyc in ROOT.rglob("*.pyc"):
    pyc.unlink(missing_ok=True)
for rel in ["outputs", "reports"]:
    (ROOT / rel).mkdir(exist_ok=True)
    (ROOT / rel / ".gitkeep").write_text("", encoding="utf-8")
print("Cleaned generated outputs, reports, caches, and bytecode files.")
