"""
extract_features.py

Builds the log-mel features (and optionally the shifted WAVs) of one condition.

  --condition plain    every WAV in <data_root>/WavFiles/ -> plain/features/logmel/{id}.pt
  --condition awgn     test split, AWGN at 15 / 5 / -5 dB  -> awgn_{snr}dB/features/logmel/{id}.pt
  --condition clutter  test split, pink noise at 15 / 5 / -5 dB -> clutter_{snr}dB/features/logmel/{id}.pt
  --condition random   1,500 pure-noise recordings          -> random_noise/features/logmel/rand_{i:08d}.pt

The noise is drawn from ONE seeded stream (torch.manual_seed(0); random.seed(0) for the random
condition), in test-CSV order and SNR order 15, 5, -5 -- exactly as in the scripts that produced the
released features. Run a condition in one go from scratch; skipping files changes the stream.

  --save_wav        also writes the noisy waveform as 32-bit float WAV: <condition>/wav/{id}.wav
  --verify_root R   compares every feature against R/<condition>/features/logmel/ and writes
                    verify_<condition>.csv (max |diff| per file); with --save_wav the check
                    re-reads the written WAV, so it covers WAV -> feature too.

Usage:
  python scripts/extract_features.py --condition awgn --data_root DATA --out_root OUT [--save_wav] [--verify_root DATA]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddl_conformal import config  # noqa: E402
from ddl_conformal.data import TEST_CSV, load_filtered_dataframe  # noqa: E402
from ddl_conformal.features import (AudioFeatureExtractor, add_awgn, add_clutter_noise,  # noqa: E402
                                    compute_amplitude_stats, load_wave, random_noise_waveform)

N_RANDOM = 1500
N_STATS_WAVS = 200
NOISE_FN = {"awgn": add_awgn, "clutter": add_clutter_noise}


class Writer:
    """Saves feature (+ WAV) of one output and optionally checks it against a reference feature."""

    def __init__(self, out_root: Path, verify_root, save_wav: bool):
        self.out_root, self.verify_root, self.save_wav = Path(out_root), verify_root, save_wav
        self.extractor = AudioFeatureExtractor(sample_rate=config.SAMPLE_RATE)
        self.log = []

    def __call__(self, cond: str, stem: str, waveform: torch.Tensor):
        feat_dir = config.logmel_dir(self.out_root, cond)
        feat_dir.mkdir(parents=True, exist_ok=True)
        if self.save_wav:
            wav_path = self.out_root / cond / "wav" / f"{stem}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_path), waveform.T.numpy(), config.SAMPLE_RATE, subtype="FLOAT")
            waveform = load_wave(wav_path)        # features come from the file that is released
        feat = self.extractor.extract_all(waveform)["logmel"].cpu()
        torch.save(feat, feat_dir / f"{stem}.pt")
        if self.verify_root:
            ref_path = config.logmel_dir(self.verify_root, cond) / f"{stem}.pt"
            diff = float((feat - torch.load(ref_path, weights_only=True)).abs().max()) if ref_path.exists() else float("nan")
            self.log.append(dict(condition=cond, file=stem, max_abs_diff=diff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["plain", "awgn", "clutter", "random"])
    ap.add_argument("--data_root", default=str(config.DATA_ROOT), help="holds annotations/ and WavFiles/")
    ap.add_argument("--out_root", default=None, help="default: --data_root")
    ap.add_argument("--save_wav", action="store_true")
    ap.add_argument("--verify_root", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N recordings")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root or data_root)
    wav_root = config.wav_dir(data_root, "plain")
    write = Writer(out_root, args.verify_root, args.save_wav and args.condition != "plain")
    torch.manual_seed(0)

    if args.condition == "plain":
        for wav_path in tqdm(sorted(wav_root.glob("*.wav"))[:args.limit or None], desc="plain logmel"):
            write("plain", wav_path.stem, load_wave(wav_path))

    elif args.condition in NOISE_FN:
        test_df = load_filtered_dataframe(config.annotations_dir(data_root) / TEST_CSV)
        add_noise = NOISE_FN[args.condition]
        for i in tqdm(range(args.limit or len(test_df)), desc=f"{args.condition} logmel (test)"):
            stem = f"{int(test_df.iloc[i]['ID']):08d}"
            waveform = load_wave(wav_root / f"{stem}.wav")
            for snr in config.SNR_LEVELS_DB:
                write(f"{args.condition}_{snr}dB", stem, add_noise(waveform, snr))

    else:
        random.seed(0)
        mean, std, vmin, vmax = compute_amplitude_stats(wav_root, N_STATS_WAVS)
        print(f"amplitude stats: mean={mean:.6f} std={std:.6f} min={vmin:.6f} max={vmax:.6f}")
        for i in tqdm(range(args.limit or N_RANDOM), desc="random-noise logmel"):
            write("random_noise", f"rand_{i:08d}", random_noise_waveform(mean, std, vmin, vmax))

    if write.log:
        log = pd.DataFrame(write.log)
        log.to_csv(out_root / f"verify_{args.condition}.csv", index=False)
        print(f"[verify] {len(log)} files, max |diff| {log.max_abs_diff.max():.3g}, "
              f"missing reference {int(log.max_abs_diff.isna().sum())}, > 1e-5: {int((log.max_abs_diff > 1e-5).sum())}")


if __name__ == "__main__":
    main()
