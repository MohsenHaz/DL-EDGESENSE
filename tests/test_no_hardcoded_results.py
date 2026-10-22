import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("chk", pathlib.Path(__file__).parents[1] / "scripts/check_no_hardcoded.py")
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)

def test_no_result_literals_in_code():
    assert chk.scan(str(pathlib.Path(__file__).parents[1])) == []

def test_withdrawn_baselines_absent_from_code():
    root = pathlib.Path(__file__).parents[1]; bad = ("EfficientSensing", "Federated-Sense", "MultiModal-Net")
    for p in list(root.rglob("*.py")) + list(root.rglob("*.yaml")):
        if "tests" in p.parts or "legacy_unverified_figures" in p.parts or p.name == "generate_figures.py": continue
        assert not any(b in p.read_text(errors="ignore") for b in bad), p
