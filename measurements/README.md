# measurements/ - raw INA219 power traces

**Status of this archive: this directory contains no traces.** They must be recorded on the physical RPi 4B / Jetson Nano
set-up and added before any energy number in the manuscript can be regenerated. Nothing in this package fabricates them:
`src/measurement/ina219_logger.py` raises if the sensor is absent, and `scripts/generate_tables.py` raises
`MissingMeasurementError` if `results/energy_summary.csv` does not exist.

## Required content
| path | content | feeds |
|---|---|---|
| `<platform>/idle/*.csv` | >= 30 s idle baseline per platform per session | idle subtraction |
| `<platform>/<mode>/session_001..050.csv` | 50 sessions x 100 inferences per operating mode | Tables 3, 4, 8, 12 |
| `<platform>/run24h/*.csv` | 24-hour run per method (DL-EgSense, DeepSense, EdgeML) | Table 7, Table 8 power rows |
| `baselines/<name>/session_*.csv` | every retained baseline, same protocol | Tables 3, 4 |
| `ablations/<variant>/session_*.csv` | every Table 19 variant, same protocol | Table 19 |

Trace format: `# effective_rate_hz=...` header lines, then `t_s,v_bus_V,i_A,gpio` (gpio = 1 during an inference).
List every file in `MANIFEST.csv`. Method: `integration_method.md`. Wiring and thermal protocol: `hardware_setup.md`.
