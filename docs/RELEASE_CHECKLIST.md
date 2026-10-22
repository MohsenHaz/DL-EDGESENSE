# Release checklist (physical steps)
1. Record raw INA219 sessions on the RPi 4B and Jetson Nano (`scripts/run_energy_profiling.py`), add them to `measurements/` and `MANIFEST.csv`, run `make reproduce`.
2. Deposit `measurements/`, `results/`, checkpoints and this package on Zenodo; put the DOI in `CITATION.cff` and in the manuscript's Data/Code Availability statements.
3. Push this package to the GitHub repository (replacing the earlier scripts that assigned results) and tag the release.
4. Run `make test verify` on a clean machine and keep the log with the release.
