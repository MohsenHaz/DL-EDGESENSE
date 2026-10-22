import subprocess, sys, pathlib, pytest
def test_reported_tables_internally_consistent():
    """Fails while the manuscript tables contradict each other (e.g. Table 4 vs Tables 8/12 for the same operating point)."""
    r = subprocess.run([sys.executable, "scripts/verify_numbers.py"], cwd=pathlib.Path(__file__).parents[1], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:]
