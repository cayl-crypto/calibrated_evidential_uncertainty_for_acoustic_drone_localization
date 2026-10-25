"""
features.py

Log-mel features and the synthetic test-time shifts (from feature_extractor.py and
kdm/extract_{plain_logmel_full,awgn_test_logmel,clutter_test_logmel,random_noise_test_logmel}.py,
same math).

  logmel   |STFT| (n_fft 2048, hop 320, Hann window 1024) -> 128 mel bands -> log1p; (8, 128, 31)
           per 8-channel 96 kHz recording of 9,600 samples
  AWGN     white Gaussian noise; SNR measured over all channels and samples together
  clutter  pink (1/f) noise, independent per channel, unit power per channel before scaling to
           the target SNR
  random   pure Gaussian noise with the pooled mean/std of 200 real recordings, clipped to their
           min/max (no drone signal, no labels)
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

from .config import AUDIO_LENGTH, NUM_CHANNELS, SAMPLE_RATE

EPS = 1e-12


class AudioFeatureExtractor:
    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=320, win_length=1024, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.mel_transform = T.MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1)

    def extract_stft(self, audio):
        window = torch.hann_window(self.win_length, device=audio.device)
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=window, return_complex=True)
        return stft.abs(), stft.angle()

    def compute_logmel(self, magnitude):
        return torch.log1p(self.mel_transform(magnitude))

    def extract_all(self, audio):
        mag, _ = self.extract_stft(audio)
        return {"logmel": self.compute_logmel(mag)}


def load_wave(path: Path) -> torch.Tensor:
    """(8, 9600) float waveform; zero-padded or cut to AUDIO_LENGTH."""
    waveform, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected SR {SAMPLE_RATE}, got {sr} for {path}")
    if waveform.shape[0] != NUM_CHANNELS:
        raise ValueError(f"Expected {NUM_CHANNELS} channels, got {waveform.shape[0]} for {path}")
    num_samples = waveform.shape[1]
    if num_samples < AUDIO_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, AUDIO_LENGTH - num_samples))
    elif num_samples > AUDIO_LENGTH:
        waveform = waveform[:, :AUDIO_LENGTH]
    return waveform


def add_awgn(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """waveform: (C, T). Adds per-call random AWGN scaled to target SNR (dB)."""
    signal_power = waveform.pow(2).mean()
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise


def generate_pink_noise(num_channels: int, length: int, device) -> torch.Tensor:
    """(C, T) unit-power pink (~1/f) noise, independent draw per channel."""
    white = torch.randn(num_channels, length, device=device)
    spectrum = torch.fft.rfft(white, dim=-1)
    freqs = torch.fft.rfftfreq(length, device=device).clamp(min=1.0 / length)  # avoid /0 at DC
    spectrum = spectrum / torch.sqrt(freqs)
    pink = torch.fft.irfft(spectrum, n=length, dim=-1)
    pink = pink / pink.std(dim=-1, keepdim=True).clamp(min=EPS)
    return pink


def add_clutter_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """waveform: (C, T). Adds per-call random pink noise scaled to target SNR (dB)."""
    C, T_ = waveform.shape
    noise = generate_pink_noise(C, T_, waveform.device)
    signal_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp(min=EPS)
    snr_linear = 10.0 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear
    noise = noise * torch.sqrt(target_noise_power / noise_power)
    return waveform + noise


def compute_amplitude_stats(wav_root: Path, n_wavs: int = 200):
    """(mean, std, min, max) pooled over n_wavs real recordings drawn with random.sample (seeded by caller)."""
    all_wavs = sorted(Path(wav_root).glob("*.wav"))
    sample_paths = random.sample(all_wavs, min(n_wavs, len(all_wavs)))
    chunks = []
    for p in sample_paths:
        try:
            chunks.append(load_wave(p))
        except Exception:
            continue
    if not chunks:
        raise RuntimeError("Could not load any real WAVs to compute amplitude stats.")
    stacked = torch.cat([c.reshape(-1) for c in chunks])
    return stacked.mean().item(), stacked.std().item(), stacked.min().item(), stacked.max().item()


def random_noise_waveform(mean: float, std: float, vmin: float, vmax: float) -> torch.Tensor:
    waveform = torch.randn(NUM_CHANNELS, AUDIO_LENGTH) * std + mean
    return waveform.clamp(min=vmin, max=vmax)
