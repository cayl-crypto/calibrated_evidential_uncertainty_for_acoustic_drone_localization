# DDL shifted: synthetic test-time shifts of the DDL test set

This record holds the distribution-shift test sets used in *Variance-Aware Conformal Calibration for Evidential Acoustic Drone Localization*. Models are trained and calibrated on clean audio only (record **DDL clean**) and then evaluated here, using the clean calibration factors. Source recordings:

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

## Conditions

All conditions except random noise are built from the 11,628 clean **test** recordings. Labels are those of the clean recording, taken from `annotations/test_dataset_with_nondrone_with_9600_flag.csv` in DDL clean.

| Condition | Folder | How it is made | Recordings |
|---|---|---|---|
| AWGN 15 / 5 / −5 dB | `awgn_{15,5,-5}dB/` | clean waveform + white Gaussian noise; noise power = signal power / 10^(SNR/10), with signal power averaged over all 8 channels and samples | 11,628 each |
| Clutter 15 / 5 / −5 dB | `clutter_{15,5,-5}dB/` | clean waveform + synthetic pink (1/f) noise, drawn independently per channel and scaled to the same SNR definition | 11,628 each |
| Random noise | `random_noise/` | pure Gaussian noise with the pooled mean and standard deviation of 200 real recordings, clipped to their min/max; no drone signal, no labels | 1,500 |

Each folder contains:
- `wav/{ID:08d}.wav` (`rand_{i:08d}.wav` for random noise): the noisy waveform, 8 channels, 96 kHz, 9,600 samples, **32-bit float** (stored without rounding).
- `features/logmel/{ID:08d}.pt`: its log-mel features, float32 `(8, 128, 31)`, computed exactly as for DDL clean.

## Reproducibility

The noise is drawn from one seeded stream (`torch.manual_seed(0)`; `random.seed(0)` too for random noise). Within it, recordings follow test-CSV order and the SNRs follow the order 15, 5, −5 dB. The command is `scripts/extract_features.py --condition {awgn,clutter,random} --save_wav` from the code record.

The log-mel features here are the ones used for every result in the paper. The WAV files were regenerated from the seed, and the features were then recomputed from the written WAV files and compared with the features used in the paper. `verification.tar.gz` lists the largest absolute difference for every file; all differences are at float32 rounding level (at most 1.3e-6 across the 69,768 AWGN and clutter files). Random-noise features have no earlier reference to compare against.

## Files

`awgn_15dB.tar`, `awgn_5dB.tar`, `awgn_-5dB.tar`, `clutter_15dB.tar`, `clutter_5dB.tar`, `clutter_-5dB.tar`, `random_noise.tar`, `verification.tar.gz`, `SHA256SUMS`.

Unpack them into the same folder as DDL clean:

```bash
for f in *.tar; do tar -xf "$f" -C DDL; done
```
