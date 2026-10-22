"""Fail if result literals are assigned in code (used by `make verify` and tests)."""
import re, sys, pathlib
BAD = re.compile(r"(?<![\w.])(51\.0|54\.1|92\.7|146\.0|138\.7|111\.0|128\.2|32\.5|24\.6|18\.7|91\.8|\b115\b)(?![\w.])")
SKIP = {"tests", "paper_artifacts", "results", "measurements", "docs", "supplementary", ".git"}
def scan(root="."):
    hits = []
    for p in pathlib.Path(root).rglob("*.py"):
        if SKIP & set(p.parts) or p.name == "check_no_hardcoded.py": continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            code = line.split("#")[0]
            if BAD.search(code): hits.append(f"{p}:{i}: {line.strip()[:90]}")
    return hits
if __name__ == "__main__":
    h = scan(); print("\n".join(h) or "OK: no result literals in code"); sys.exit(1 if h else 0)
