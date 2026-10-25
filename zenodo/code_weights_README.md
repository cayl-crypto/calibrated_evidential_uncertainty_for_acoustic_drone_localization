# Model weights and per-sample results

These are the trained models and per-recording outputs behind *Variance-Aware Conformal Calibration for Evidential Acoustic Drone Localization*. The `README.md` of the code repository explains how to use these files. The data is in the records **DDL clean** and **DDL shifted**.

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

## Files

| File | Unpack in the code folder to | Contents |
|---|---|---|
| `weights.tar` | `weights/` | `evidential/<hp>/` for each of the 16 settings: `model.pt` (state_dict + config), `conformal_q_{sin,cos,range}_{6827,9545,9973}.pt`, `run_summary.csv`, `val_epoch_csv/`. `base_model/`: `model.pt`, `conformal_q_{sin,cos,range,bearing,euclidean}_{6827,9545,9973}.pt`, `val_epoch_csv/`. Plus `conversion_check.csv`. |
| `per_sample.tar.gz` | `results/final_seed12321/per_sample/` | `evidential/<hp>/{plain,plain_val,awgn_15dB,awgn_5dB,awgn_-5dB,clutter_15dB,clutter_5dB,clutter_-5dB}.csv` and `base_model/plain.csv` |
| `SHA256SUMS` | | checksums |

```bash
cd <code folder>
tar -xf weights.tar
tar -xzf per_sample.tar.gz
```

`<hp>` is `c{c}_kl{eta}` with `.` written as `p`. For example, `c0p7_kl0p01` means evidence-regularizer coefficient c = 0.7 and external-prior KL weight η = 0.01; this is the selected model.

## Notes

- All 16 models were trained on clean audio with seed 12321. The Base Model was trained once without a fixed seed.
- The q-hat values come from the clean validation calibration split (2,912 recordings), and the same clean q-hat is used on every test condition.
- The original checkpoints stored the whole pickled module. They were converted to state_dicts, and the outputs of the converted and original models were compared on real test features: they are identical (`conversion_check.csv`, maximum difference 0).
- Each per-sample CSV row is one recording with these columns:
  - targets `y_*`;
  - means `mu_*`;
  - NIG parameters `v_*`, `alpha_*`, `beta_*`;
  - predictive `sigma_*`;
  - the raw ±1σ interval `L/U/inside/W_*_base`;
  - conformal interval coverage and widths at each level, `inside_*_q{tag}` and `W_*_q{tag}`.

  Base Model files use `pred_*`, `gt_*`, `err_*`, `cov_*_q{tag}` and `q_*_{tag}`.
