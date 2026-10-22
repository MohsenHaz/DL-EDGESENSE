"""Read helpers. Everything the table/figure scripts need comes from results/ or measurements/ - never from literals."""
import csv, json, glob, os


class MissingMeasurementError(RuntimeError):
    pass


def load_energy_summary(path="results/energy_summary.csv"):
    if not os.path.exists(path):
        raise MissingMeasurementError(
            f"{path} not found. Run `make aggregate` after adding raw traces to measurements/ (see measurements/README.md).")
    with open(path) as f:
        return list(csv.DictReader(f))


def energy_mj(summary, platform, config, stat="energy_mj_mean"):
    for r in summary:
        if r["platform"] == platform and r["config"] == config:
            return float(r[stat])
    raise MissingMeasurementError(f"no energy row for platform={platform}, config={config}")


def load_json_dir(path):
    files = sorted(glob.glob(os.path.join(path, "*.json")))
    if not files:
        raise MissingMeasurementError(f"no result files in {path}")
    out = []
    for f in files:
        with open(f) as fh:
            out.append(json.load(fh))
    return out
