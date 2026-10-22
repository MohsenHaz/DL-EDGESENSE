# Reproduction guide (v3)
1. `pip install -r requirements.txt` (PyTorch 2.1.0, TFLite 2.14.0); download datasets with `python data/download_datasets.py` (MMAct, EnviroSense, SpectrumSense, DeepSense 6G).
2. Train: `python scripts/train.py --config configs/standard_mode.yaml` (repeat per mode and seed).
3. Evaluate F1 (Table S1 / Table 4): `python scripts/evaluate.py --config ... --checkpoint ...`.
4. On the board: `python scripts/run_energy_profiling.py --platform rpi4b` (and `jetson_nano`); add the produced CSVs to `measurements/MANIFEST.csv`.
5. `make aggregate tables ablation`; `make figures-measured`; `make verify`.
6. Expected values are listed in `claims.yaml`; they are *checked*, never copied into code.
No expected-result numbers are stated in this guide on purpose: earlier versions quoted 51.0 / 48.2 mJ and 115 ms for different modes and platforms.
