# measurements/ - raw INA219 power traces

This directory holds the layout, manifest format and documentation for the raw INA219 traces, which are released in the Zenodo archive of the package. `src/measurement/ina219_logger.py` raises if the sensor is absent, and `scripts/generate_tables.py` raises `MissingMeasurementError` if `results/energy_summary.csv` does not exist, so all energy values come from recorded traces.

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
