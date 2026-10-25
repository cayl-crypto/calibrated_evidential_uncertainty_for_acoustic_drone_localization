# DDL clean: multi-channel drone audio, annotations and log-mel features

This record holds the clean recordings used in *Variance-Aware Conformal Calibration for Evidential Acoustic Drone Localization*. It contains the raw audio, the train/validation/test split with the calibration subset, and the log-mel features the models read. The recordings come from the DDL dataset:

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

The shifted test sets (AWGN, clutter, random noise) are a separate record, **DDL shifted**. The code, model weights and per-sample results are in the code record. All three are linked in the related identifiers.

## Files

| File | Contents | Unpacks to |
|---|---|---|
| `annotations.tar.gz` | train / val / test CSVs | `annotations/` |
| `WavFiles.tar` | 58,136 recordings, `{ID:08d}.wav`: 8 channels, 96 kHz, 9,600 samples (0.1 s), 32-bit integer PCM | `WavFiles/` |
| `plain_logmel.tar` | 58,136 log-mel tensors, `{ID:08d}.pt`, float32 `(8, 128, 31)` | `plain/features/logmel/` |
| `SHA256SUMS` | checksums | |

Unpack everything, together with the DDL shifted record, into one folder. That folder is the code's `DDL_DATA_ROOT`:

```bash
mkdir -p DDL && for f in *.tar *.tar.gz; do tar -xf "$f" -C DDL; done
sha256sum -c SHA256SUMS
```

## Splits

Only rows with `is_corrupt == False` and `sample_9600 == True` are used; that's every row in these CSVs.

| Split | CSV | Rows | Use |
|---|---|---|---|
| train | `train_dataset_with_nondrone_with_9600_flag.csv` | 40,692 | training |
| validation, evaluation | `val_dataset_with_nondrone_with_9600_flag.csv`, `conformal_role == "evaluation"` | 2,904 | early stopping, hyper-parameter selection |
| validation, calibration | same CSV, `conformal_role == "calibration"` | 2,912 | conformal calibration (q-hat) |
| test | `test_dataset_with_nondrone_with_9600_flag.csv` | 11,628 | evaluation |

## Annotation columns used

| Column | Meaning |
|---|---|
| `ID` | recording ID; file name `{ID:08d}.wav` / `.pt` |
| `Sample Class` | drone type: `MINI` or `PRO4` |
| `Bearing to Target` | bearing in degrees, 0–359 (targets: sin and cos of it) |
| `Range to Target` | range in metres, 1–249 (target: range / 250) |
| `is_corrupt`, `sample_9600` | quality filters (see above) |
| `conformal_role` | validation only: `evaluation` or `calibration` |

The other columns are carried over from the source dataset and aren't used in this work.

## Log-mel features

For each recording (all 8 channels):
1. STFT with n_fft 2048, hop 320, Hann window of 1,024.
2. Magnitude.
3. 128 mel bands at 96 kHz.
4. `log1p`.

Recordings shorter than 9,600 samples are zero-padded. Code: `ddl_conformal/features.py`; to rebuild, run `scripts/extract_features.py --condition plain`.
