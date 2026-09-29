DL-EgSense reproducibility package (updated 27 Sep 2026, patch round 2)

WHAT'S NEW IN THIS PATCH
  Previous package was missing the actual model/training/data code
  referenced by Supplementary S5. This patch adds:
    - src/models/          EACMA (eacma.py), HPEQ (hpeq.py), feature
                            extractors, fusion engine, full model assembly
    - src/losses/          Eq (15)-(18) energy-aware training objective
    - src/data/loaders/     MMAct, EnviroSense, SpectrumSense, DeepSense6G
    - src/data/sync/        unified-clock drift-correction buffer
    - src/utils/            logger.py, hardware_profiler.py (previously
                            imported but missing -- this caused
                            ModuleNotFoundError in benchmark.py and
                            energy_profiler.py; now fixed)
    - src/training/early_stopping.py   (was an empty stub; implemented)
    - configs/              training.yaml, loss.yaml, quantization_hpeq.yaml,
                            one yaml per operating mode, platforms/*.yaml
    - train.py, preprocess.py   the two entry points S5 already told
                            readers to run
    - requirements.txt      pinned to Table S3 (PyTorch 2.1.0, TF-Lite 2.14.0)
    - tests/                regression test proving Eq (9) attention
                            weights sum to 1, and that Eq (4) quantiser
                            is a function of both x and b

WHAT STILL REQUIRES THE USER
  - The four datasets themselves are NOT bundled (respect each
    dataset's own license/hosting) -- download via the URLs in the
    manuscript's Data Availability section and point --raw-root at them.
  - src/utils/hardware_profiler.py talks to a real INA219 sensor; on a
    dev machine without one attached it falls back to a clearly-logged
    SIMULATION mode. Only readings taken on the physical RPi 4B /
    Jetson Nano with INA219 wired in reproduce Tables 3/4/8/12/13/17/19.
  - Video decoding in the MMAct/DeepSense6G loaders is left as a
    placeholder (returns a zero tensor) so the pipeline is runnable
    end-to-end without pinning a specific video backend; plug in
    torchvision.io / decord / opencv per your environment.
  - This sandbox could not run a live `pip install torch` (disk-space
    and broken CUDA-lib issues in the container used to prepare this
    patch), so `pytest tests/` should be re-run in your own environment
    before you treat it as passing. The Eq (9) and Eq (4) logic was
    independently verified in pure NumPy (see patch notes) and is
    algebraically identical to the PyTorch implementation.

CONTENTS (unchanged from previous package)
  DL-EgSense-V4-Revised.docx / .pdf   (superseded by V5 -- see below)
  DL-EgSense_Supplementary_Material_V2.docx
  figures/                            13 manuscript figures
  scripts/fig4_energy_accuracy_tradeoff.py
  DL-EDGESENSE-repo/                  now contains the full source tree
                                       described above

NOTE: DL-EgSense-V5-Revised.docx (the latest manuscript, with the
Table 4 baseline-reference fix, Abstract/Table 2 alignment, and Figure 4
regeneration) is provided alongside this package but was not part of
the original zip -- copy it in before re-uploading to GitHub if you
want the repo and the manuscript to stay in lockstep.
