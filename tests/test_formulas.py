import pytest
def saved(e_abl, e_full): return (e_abl - e_full) / e_abl * 100        # Table 19 "Energy Saved"
def change(e_abl, e_full): return (e_full - e_abl) / e_full * 100      # Table 19 "Energy Change"
def savings(e, ref): return (ref - e) / ref * 100

def test_table19_formulas():
    assert saved(78.3, 54.1) == pytest.approx(30.9, abs=0.05)
    assert change(78.3, 54.1) == pytest.approx(-44.7, abs=0.05)
    assert saved(94.8, 54.1) == pytest.approx(42.9, abs=0.05)

def test_savings_vs_baselines():
    assert savings(54.1, 146.0) == pytest.approx(62.9, abs=0.05)
    assert savings(51.0, 146.0) == pytest.approx(65.1, abs=0.05)       # the arithmetic slip fixed in the manuscript
    assert savings(54.1, 91.2) == pytest.approx(40.7, abs=0.05)

def test_weighted_f1_table_s1():
    f1 = [91.7, 92.3, 93.0, 94.3]; w = [0.32, 0.24, 0.20, 0.24]
    assert sum(a * b for a, b in zip(f1, w)) == pytest.approx(92.7, abs=0.15)
