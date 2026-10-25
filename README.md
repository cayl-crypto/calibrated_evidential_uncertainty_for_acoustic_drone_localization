# Variance-Aware Conformal Calibration for Evidential Acoustic Drone Localization

This repository holds the code, results and instructions for reproducing the paper. A Conformer encoder reads 8-channel log-mel features and predicts the drone's bearing (as sin and cos) and range. Each output has a Normal-Inverse-Gamma (NIG) head trained with an evidential loss that includes an external-prior KL term. The evidential predictive σ is then calibrated per output with split conformal prediction, which gives intervals with guaranteed coverage at 68.27 / 95.45 / 99.73%. The intervals of the three outputs combine into a spatial region (a "keyhole") on the range–bearing plane.

The audio comes from the DDL dataset:

```bibtex
@inproceedings{safavi2025ddl,
  title={Ddl: A dataset for drone detection and localization from multi-channel audio and a deep uncertainty-aware framework},
  author={Safavi, Saeid and {\c{C}}ayli, {\"O}zkan and Safavi, Mohammad Amin and Cook, Ben and Wang, Wenwu},
  booktitle={2025 IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={01--06},
  year={2025},
  organization={IEEE}
}
```

---

## Contents

1. [Method](#1-method)
2. [Repository layout](#2-repository-layout)
3. [Zenodo records](#3-zenodo-records)
4. [Installation and data](#4-installation-and-data)
5. [Reproducing the paper](#5-reproducing-the-paper)
6. [Using the released models](#6-using-the-released-models)
7. [Where each table and figure comes from](#7-where-each-table-and-figure-comes-from)
8. [Reproducibility notes](#8-reproducibility-notes)
9. [Preparing and uploading the Zenodo records (maintainers)](#9-preparing-and-uploading-the-zenodo-records-maintainers)

---

## 1. Method

### Inputs and targets
- **Input:** a 0.1 s recording (8 channels, 96 kHz, 9,600 samples), turned into a log-mel tensor of shape `(8, 128, 31)`:
  - STFT with n_fft 2048, hop 320, Hann window of 1,024;
  - 128 mel bands;
  - `log1p`.
- **Targets:**
  - sin θ and cos θ of the bearing θ;
  - range r / 250 m.

### Model
`ddl_conformal/models.py`:
- `Conformer8Mic`: two 3×3 convolution layers (128 channels), a linear projection to 192 dimensions, a 6-layer Conformer (6 heads), then mean pooling over time.
- **Evidential model:** three linear heads, one per output, each giving (μ, v, α, β).
  - μ is bounded with tanh for sin/cos and sigmoid for range.
  - v, α, β pass through softplus inside the loss: v = softplus + ε, α = softplus + 1 + ε, β = softplus + ε, with ε = 1e-6.
- **Base Model:** the same encoder with one linear head (tanh, tanh, sigmoid), trained with MSE.

### Loss (`ddl_conformal/losses.py`, `EvidentialNIGLossExternalPrior`)
For each output:

```
L = NLL_NIG(y | μ, v, α, β)  +  c · |y − μ| · (2v + α)  +  η · |y − μ| · KL( NIG(μ, v, α, β) ‖ NIG(0, 1, 1.5, 1) )
```

- **The first two terms are the reference evidential regression loss** (Amini et al., 2020). The error |y − μ| isn't detached. With η = 0 the loss is exactly that reference loss (plus ε in the activation).
- **The third term is the KL divergence to a fixed external prior NIG(μ₀ = 0, v₀ = 1, α₀ = 1.5, β₀ = 1),** in exact form:
  - v-terms ½(log(v/v₀) + v₀/v);
  - the μ-term scaled by E[1/σ²] = α/β;
  - the Gamma-part terms.

  It is multiplied by the error, following the reference's `reg = error · kl`.
- **In code and file names,** c is `coeff` and η is `kl_weight`.

The losses of the three heads are summed.

### Training
`ddl_conformal/training.py`:
- **Optimiser and schedule:** AdamW (lr 5e-5), cosine schedule (T_max 20), batch 64.
- **Precision and clipping:** mixed precision on GPU, gradient clipping at 1.0.
- **Early stopping:** patience 3 on the validation evaluation split. The evidential model stops on total Student-t NLL; the Base Model on MSE.
- **Data:** clean audio only. The final grid is c ∈ {0.05, 0.1, 0.3, 0.7} × η ∈ {0, 0.01, 0.05, 0.2}, seed 12321.

### Conformal calibration
`ddl_conformal/conformal.py`, using the clean validation **calibration** split (2,912 recordings), per output and per level (α = 0.3173, 0.0455, 0.0027):

- **Evidential + Conformal:**
  - score s = |y − μ| / σ, with σ² = β(1 + v) / (v(α − 1));
  - interval clip(μ ± q̂σ).
- **Base Model + Conformal:**
  - score s = |y − ŷ| (range in metres);
  - interval clip(ŷ ± q̂).
- **Raw NIG ("Evidential"):** μ ± σ.
- **q̂** is the ⌈(n + 1)(1 − α)⌉-th smallest score.
- **Clipping:** intervals are clipped to [−1, 1] for sin/cos and [0, 1] for range. All metrics use the actual clipped bounds L and U.
- **Shifted test sets** reuse the clean q̂; nothing is recalibrated.

### Hyper-parameter selection
On the validation **evaluation** split (2,904 recordings, disjoint from calibration and test):
1. A setting passes if its sin, cos and range conformal coverage at 95.45% are each ≥ 95.45%.
2. The passing settings are ranked by Euclidean MAE.

Rank 1 (★) is **c = 0.7, η = 0.01**; ranks 2 and 3 are c 0.7 / η 0.2 and c 0.1 / η 0. Only 3 of the 16 settings pass.

### Metrics
`analysis/generate_all_results.py`:
- **Component coverage:** share of recordings with y inside [L, U], per output.
- **MWIS:** per recording, the Winkler scores (U − L) + (2/α)·(L − y)₊ + (2/α)·(y − U)₊ are summed over sin, cos and range (normalised units); MWIS is the mean of this sum over recordings.
- **Spatial ("keyhole") coverage (SC):** the ground truth lies inside all three intervals.
  - S+R¬C: inside sin and range but not cos.
  - C+R¬S: inside cos and range but not sin.
- **Empty region:** the sin and cos intervals admit no bearing at all (no θ with sin θ ∈ [L_s, U_s] and cos θ ∈ [L_c, U_c]).
- **Area A_pred:** the annular sector between the range bounds, over the bearings admitted by the sin/cos intervals. Bearings are checked on a 720-point grid. The area is also given as % of the disk π·250².
- **Localization errors:**
  - bearing error, wrapped to (−180°, 180°];
  - range error in metres;
  - Euclidean distance between the predicted and true positions.

### Test-time shifts
`ddl_conformal/features.py`, test split only:
- **AWGN** at 15 / 5 / −5 dB.
- **Clutter:** synthetic pink (1/f) noise at the same SNRs.
- **Random noise:** pure noise, 1,500 recordings, unlabelled; not used in the paper's tables.

SNR is computed against the recording's power over all channels and samples.

---

## 2. Repository layout

```
release_external_prior/
├── ddl_conformal/                  library
│   ├── config.py                   paths (env vars / flags), grid, constants
│   ├── data.py                     splits and dataloaders over log-mel features
│   ├── features.py                 log-mel extractor, AWGN / clutter / random-noise generators
│   ├── losses.py                   reference ERN port + EvidentialNIGLossExternalPrior
│   ├── models.py                   Conformer8Mic, evidential model, Base Model, checkpoint I/O
│   ├── training.py                 training loops
│   ├── conformal.py                conformal scores and q-hat
│   └── evaluation.py               validation metrics, per-sample result tables
├── scripts/
│   ├── extract_features.py         features (+ shifted WAVs) per condition
│   ├── run_evidential.py           one grid setting: train → calibrate → evaluate (clean, AWGN, clutter, validation)
│   ├── run_base.py                 Base Model: train → calibrate → evaluate (clean)
│   └── slurm/                      example SLURM scripts (16-task array, base model, features)
├── analysis/
│   ├── generate_all_results.py     every table and figure of the paper from the per-sample results
│   └── generate_polar_qualitative_per_case.py   per-case polar figures
├── tools/
│   ├── package_zenodo.sh           builds the upload folders of the three Zenodo records
│   ├── upload_to_zenodo.py         uploads a folder to a Zenodo draft (never publishes)
│   └── convert_release_artifacts.py one-time conversion of the original run outputs (provenance)
├── zenodo/                         README and metadata of each Zenodo record
├── results/final_seed12321/        in git (small)
│   ├── tables/  all_settings/  figures/  overleaf/
│   ├── qhat_table.csv              every q-hat (16 evidential models × 3 outputs × 3 levels + Base Model)
│   ├── test_sample_class.csv       drone class per test recording
│   └── per_sample/                 ← Zenodo (gitignored), 1.7 GB
└── weights/                        ← Zenodo (gitignored), 556 MB
    ├── evidential/<hp>/            model.pt, conformal_q_*.pt, run_summary.csv, val_epoch_csv/
    └── base_model/                 model.pt, conformal_q_*.pt, val_epoch_csv/
```

`<hp>` is `c{c}_kl{η}` with `.` written as `p`; for example, `c0p7_kl0p01` is the selected model.

---

## 3. Zenodo records

| Record | Contents | Size |
|---|---|---|
| **DDL clean** | annotations (splits, calibration subset), 58,136 clean WAVs, clean log-mel features | ~24 GB |
| **DDL shifted** | AWGN and clutter at 15/5/−5 dB (11,628 test recordings each) and 1,500 random-noise recordings: 32-bit float WAVs + log-mel features | ~31 GB |
| **Weights and results** | `weights.tar` (16 evidential models + Base Model, q-hat files) and `per_sample.tar.gz` | ~1 GB |

The large files are listed in `.gitignore` (`weights/`, `results/*/per_sample/`, `data/`). Git therefore only holds code, the README, the summary tables and the figures. Each record's own README is in `zenodo/`.

---

## 4. Installation and data

```bash
python -m venv venv && source venv/bin/activate      # Python 3.10 used for the paper; 3.8+ works
pip install -r requirements.txt
```

### Data
Download DDL clean and DDL shifted, then unpack both into **one** folder:

```bash
mkdir -p data
for f in DDL_clean/*.tar* DDL_shifted/*.tar*; do tar -xf "$f" -C data; done
```

This gives:

```
data/annotations/                 train / val / test CSVs
data/WavFiles/{ID}.wav            clean audio
data/plain/features/logmel/       clean features (train, val, test)
data/awgn_{15,5,-5}dB/{wav,features/logmel}/
data/clutter_{15,5,-5}dB/{wav,features/logmel}/
data/random_noise/{wav,features/logmel}/
```

`data/` inside this folder is the default. Otherwise set `export DDL_DATA_ROOT=/path/to/data` or pass `--data_root`. Weights and per-sample results default to `weights/` and `results/final_seed12321/`; override them with `DDL_RUNS_ROOT` / `--runs_root` and `DDL_RESULTS_ROOT` / `--results_root`.

### Released weights and per-sample results

```bash
tar -xf weights.tar                # -> weights/
tar -xzf per_sample.tar.gz         # -> results/final_seed12321/per_sample/
```

---

## 5. Reproducing the paper

Every step skips work whose outputs already exist. Paths come from §4.

### Step 0: features (optional; the records already contain them)

```bash
python scripts/extract_features.py --condition plain
python scripts/extract_features.py --condition awgn    [--save_wav]
python scripts/extract_features.py --condition clutter [--save_wav]
python scripts/extract_features.py --condition random  [--save_wav]
```

The noise is drawn from one seeded stream. Run each condition in one go: skipping files changes the stream.

### Step 1: train, calibrate and evaluate the 16 evidential settings

On one GPU, per setting (`--job_index` 0–15, in coeff-major order; 13 is the selected model):

```bash
python scripts/run_evidential.py --job_index 13
# or: python scripts/run_evidential.py --coeff 0.7 --kl_weight 0.01
```

All 16 as a SLURM array: `sbatch scripts/slurm/run_evidential_grid.sh` (edit the partition / environment lines first). Each task uses one GPU (the final runs stopped after 21–24 epochs) and writes:
- `weights/evidential/<hp>/`: model, q-hat, training log;
- `results/final_seed12321/per_sample/evidential/<hp>/`: `plain.csv` (clean test), `awgn_*.csv`, `clutter_*.csv`, and `plain_val.csv` (validation evaluation split, used for model selection).

### Step 2: Base Model + Conformal

```bash
python scripts/run_base.py          # or sbatch scripts/slurm/run_base.sh
```

### Step 3: tables and figures

CPU only, a few minutes:

```bash
python analysis/generate_all_results.py --all_settings
python analysis/generate_polar_qualitative_per_case.py
```

These write `results/final_seed12321/{tables,figures,all_settings}/`. To get only the tables and figures from the released results, skip steps 0–2: unpack `per_sample.tar.gz` and run step 3. `qhat_table.csv` is in git, so the weights aren't needed for this.

---

## 6. Using the released models

```python
import torch
from ddl_conformal.models import load_checkpoint
from ddl_conformal.conformal import load_qhats, nig_predictive_sigma_from_pack

model = load_checkpoint("weights/evidential/c0p7_kl0p01/model.pt")      # selected model
q = load_qhats("weights/evidential/c0p7_kl0p01")                         # {"6827"|"9545"|"9973": {"sin","cos","range": q̂}}

x = torch.load("data/plain/features/logmel/00007341.pt").unsqueeze(0)   # (1, 8, 128, 31)
with torch.no_grad():
    (mu_sin, mu_cos, mu_rng), packs = model.forward_all(x)
for name, mu, pack, (lo, hi) in zip(["sin", "cos", "range"], (mu_sin, mu_cos, mu_rng), packs,
                                    [(-1, 1), (-1, 1), (0, 1)]):
    sigma = nig_predictive_sigma_from_pack(pack)
    L, U = (mu - q["9545"][name] * sigma).clamp(lo, hi), (mu + q["9545"][name] * sigma).clamp(lo, hi)
    print(name, float(mu), (float(L), float(U)))       # range is normalised: multiply by 250 m
```

The Base Model works the same way: `load_checkpoint("weights/base_model/model.pt")` returns (sin, cos, range). Its q-hat files (`load_qhats(dir, ["sin", "cos", "range", "bearing", "euclidean"])`) are absolute half-widths, with range in metres.

---

## 7. Where each table and figure comes from

All the files below are in `results/final_seed12321/` and are produced by `analysis/generate_all_results.py`, except where another script is named. The LaTeX used in the paper is in `overleaf/`.

| Paper | File(s) | Notes |
|---|---|---|
| Accuracy table (`tab:accuracy`) | `tables/accuracy_table.csv`, `overleaf/tab_accuracy.tex` | |
| Coverage + MWIS table (`tab:coverage`) | `tables/coverage_table.csv`, `overleaf/tab_coverage.tex` | |
| Spatial coverage table (`tab:spatial`) | `tables/spatial_coverage_table.csv`, `overleaf/tab_spatial.tex` | |
| Grid-search tables (`tab:supp:gridsearch_*`, `tab:supp:cva_data`) | `all_settings/all_settings_long.csv`, `overleaf/tab_supp_gridsearch.tex` | clean test set |
| Shift accuracy (`tab:plain_localization_accuracy`) | `all_settings/all_settings_long.csv` (hp `c0p7_kl0p01`, AWGN rows), `overleaf/tab_plain_localization_accuracy.tex` | |
| Shift coverage / area / MWIS (`tab:distributional_shift_combined`) | `tables/noise_full_metrics.csv`, `overleaf/tab_distributional_shift_combined.tex` | Evid+Conf ★, clean q-hat |
| HP selection | `tables/hp_selection.csv` | validation evaluation split |
| Coverage vs area | `figures/coverage_vs_area.png` | |
| Range–bearing coverage maps | `figures/range_bearing/`, `figures/compact_comparison/` | |
| Polar qualitative examples | `figures/polar_qualitative/` (`per_case/` from `generate_polar_qualitative_per_case.py`) | |
| Noise robustness | `figures/noise_robustness.png`, `tables/noise_coverage_table.csv` | |

---

## 8. Reproducibility notes

- **Retraining gives slightly different models.** Training is seeded (12321) but uses non-deterministic cuDNN kernels and mixed precision. Retraining changes the numbers slightly, and which settings pass the validation gate can change too. On clean data the results are stable across seeds. On the shifted test sets, single-setting results vary between seeds, while the overall behaviour holds: confidence rises as the error rises, and coverage collapses at low SNR.
- **The released weights reproduce the released results on any device.** They are the original checkpoints, converted to state_dicts with identical outputs (`weights/conversion_check.csv`). Recomputing q̂ and the per-sample results from them on a CPU matched the released (GPU) values to float32 rounding: about 1e-5 relative for q̂, and no coverage flag changed.
- **The noise is seeded:** the shifted test sets come from seeded generators. The released WAVs were regenerated from the seed, and their features match the features used in the paper to float32 rounding (`verification.tar.gz` in DDL shifted).

---

## 9. Preparing and uploading the Zenodo records (maintainers)

1. **Shifted WAVs and features,** checked against the features used in the paper:
   ```bash
   for c in awgn clutter random; do
     python scripts/extract_features.py --condition $c --data_root <DDLDataset> --out_root <SHIFTED> --save_wav --verify_root <DDLDataset>
   done
   ```
2. **Upload folders** (tars + README + SHA256SUMS):
   ```bash
   bash tools/package_zenodo.sh <DDLDataset> <SHIFTED> <OUT> all
   ```
3. **Metadata:** `zenodo/{DDL_clean,DDL_shifted,code_weights}.json` hold the title, description, creators, licence and access of each record; review them before uploading.
4. **Upload to a draft.** The script resumes and checks MD5; it never publishes.
   ```bash
   export ZENODO_TOKEN=...            # Zenodo → Account → Applications → personal token (deposit:write, deposit:actions)
   python tools/upload_to_zenodo.py --folder <OUT>/DDL_clean/upload    --metadata zenodo/DDL_clean.json
   python tools/upload_to_zenodo.py --folder <OUT>/DDL_shifted/upload  --metadata zenodo/DDL_shifted.json
   python tools/upload_to_zenodo.py --folder <OUT>/code_weights/upload --metadata zenodo/code_weights.json
   ```
   Add `--sandbox` to test on sandbox.zenodo.org first.
5. **Publish:** check each draft on zenodo.org and press Publish. This assigns the DOI and can't be undone. Then add the DOIs to the related identifiers of each record.
