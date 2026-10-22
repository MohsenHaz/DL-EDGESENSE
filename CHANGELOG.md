# Changelog
## v3 (this release) - response to review
- Removed every hard-coded result from code (ablation placeholder 51.0/92.7/115, baseline energy/F1 dictionaries, table/figure constants, `50 mW` profiler placeholder).
- Added raw-trace measurement pipeline (`src/measurement/`), manifest, integration method, `results/energy_summary.csv` generation.
- Added real ablation variants (`src/models/ablation.py`) and an aggregation step that computes the "Energy Saved" / "Energy Change" columns by formula.
- Withdrew EfficientSensing, Federated-Sense and MultiModal-Net from code, figures and tables; figure generator rejects them.
- Regenerated manuscript figures 2, 4, 5, 6, 7, 8, 9, 13 from the reported tables (`paper_artifacts/`).
- Added `claims.yaml`, `scripts/verify_numbers.py`, `scripts/check_no_hardcoded.py` and tests.
- Old figure files (which still plotted the withdrawn baselines) are kept only in `legacy_unverified_figures/` for audit.
- NOT yet included: raw traces, trained checkpoints, baseline re-implementation configs/weights, dataset files.
