import csv, numpy as np, pytest
from src.measurement import integrate_energy as ie
from src.measurement.aggregate import aggregate, ManifestError
from src.measurement.power_check import check_instantaneous_peak, check_label, implied_power_mw


def synth(path, idle_mw=500.0, extra_mw=400.0, n_inf=5, dur_s=0.1, gap_s=0.1, rate=20000):
    """SYNTHETIC trace used only to unit-test the integrator (never written under measurements/)."""
    t = np.arange(0, n_inf * (dur_s + gap_s) + 0.1, 1 / rate); g = np.zeros_like(t, dtype=int)
    for k in range(n_inf):
        a = 0.05 + k * (dur_s + gap_s); g[(t >= a) & (t < a + dur_s)] = 1
    p = (idle_mw + extra_mw * g) / 1e3; v = np.full_like(t, 5.0); i = p / v
    with open(path, "w", newline="") as f:
        f.write(f"# effective_rate_hz={rate}\n"); w = csv.writer(f); w.writerow(["t_s", "v_bus_V", "i_A", "gpio"])
        w.writerows(zip(t, v, i, g))


def test_known_energy(tmp_path):
    p = tmp_path / "s.csv"; synth(p)
    tr = ie.load_trace(p); res = ie.per_inference(tr, idle_w=0.5)
    assert len(res) == 5
    for r in res:                              # 400 mW x 0.1 s = 40 mJ, window ~100 ms
        assert r["energy_mj"] == pytest.approx(40.0, rel=0.02)
        assert r["mean_power_mw"] == pytest.approx(400.0, rel=0.02)


def test_idle_estimate(tmp_path):
    p = tmp_path / "i.csv"; synth(p, extra_mw=0.0); assert ie.idle_power_w(ie.load_trace(p)) == pytest.approx(0.5, rel=1e-3)


def test_effective_rate(tmp_path):
    p = tmp_path / "s.csv"; synth(p, rate=5000); assert ie.effective_rate_hz(ie.load_trace(p)) == pytest.approx(5000, rel=1e-2)


def test_peak_consistency_rule():
    ok, need = check_instantaneous_peak(54.1, 115, 248.5)
    assert need == pytest.approx(470.4, abs=0.1) and not ok          # the contradiction a reviewer found
    assert check_instantaneous_peak(54.1, 115, 600)[0]
    assert not check_label("Peak Power [mW, 24h run]") and check_label("Peak of windowed-mean power")


def test_aggregate_from_manifest(tmp_path):
    root = tmp_path / "m"; (root / "rpi4b").mkdir(parents=True)
    synth(root / "rpi4b/idle.csv", extra_mw=0.0); synth(root / "rpi4b/s1.csv")
    (root / "MANIFEST.csv").write_text("file,platform,config,role,session,table\nrpi4b/idle.csv,rpi4b,idle,idle,0,all\nrpi4b/s1.csv,rpi4b,standard,inference,1,8\n")
    out = aggregate(str(root), str(tmp_path / "e.csv")); rows = list(csv.DictReader(open(out)))
    assert float(rows[0]["energy_mj_mean"]) == pytest.approx(40.0, rel=0.03)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(ManifestError): aggregate(str(tmp_path), str(tmp_path / "x.csv"))
