#!/usr/bin/env python3
"""
Validate Stage-1 detector against the DroneAudioDataset.

Downloads the dataset from GitHub (with --download flag), iterates
audio files by class folder, runs process_clip on each clip, and
reports per-class detection rates.

Usage
-----
    # First time -- download the dataset:
    python scripts/validate_on_dataset.py --download

    # Subsequent runs (dataset already on disk):
    python scripts/validate_on_dataset.py --dataset-dir DroneAudioDataset

    # Custom detection threshold:
    python scripts/validate_on_dataset.py --clip-thresh 0.40
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_audio(path: str, target_sr: int = 16000):
    """Load an audio file, returning (samples, sample_rate).

    Tries soundfile first, falls back to scipy.io.wavfile.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float64')
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Normalise integer formats to [-1, 1]
        if data.dtype == np.int16 or np.max(np.abs(data)) > 1.0:
            data = data / 32768.0
        return data, sr
    except Exception as e:
        print(f"  [skip] Cannot load {path}: {e}")
        return None, None


def download_dataset(dest_dir: str):
    """Clone the DroneAudioDataset repository from GitHub."""
    url = "https://github.com/saraalemadi/DroneAudioDataset.git"
    if os.path.isdir(dest_dir):
        print(f"Dataset directory already exists: {dest_dir}")
        return
    print(f"Cloning {url} -> {dest_dir} ...")
    subprocess.check_call(["git", "clone", "--depth", "1", url, dest_dir])
    print("Download complete.")


def find_class_folders(dataset_dir: str):
    """Find subfolders that contain audio files, grouped by class name.

    Returns dict: class_name -> list of audio file paths.
    """
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    classes = defaultdict(list)

    dataset_path = Path(dataset_dir)
    for root, _dirs, files in os.walk(dataset_path):
        for f in files:
            if Path(f).suffix.lower() in audio_exts:
                # Use the immediate parent folder as class label
                class_name = Path(root).name
                classes[class_name].append(os.path.join(root, f))

    return dict(classes)


def validate(dataset_dir: str, clip_thresh: float = 0.30,
             require_specificity: bool = False):
    """Run Stage-1 detector on each clip and report per-class rates."""
    from drone_detector.detector import DroneDetector, DetectorConfig

    classes = find_class_folders(dataset_dir)
    if not classes:
        print(f"No audio files found in {dataset_dir}")
        sys.exit(1)

    cfg = DetectorConfig(require_specificity=require_specificity)
    det = DroneDetector(cfg)

    print(f"\nClip detection threshold: >{clip_thresh:.0%} of frames\n")
    print(f"{'Class':<25s} {'Clips':>6s} {'Detected':>9s} {'Rate':>7s}")
    print("-" * 50)

    for class_name in sorted(classes.keys()):
        files = sorted(classes[class_name])
        n_clips = len(files)
        n_detected = 0

        for fpath in files:
            audio, sr = load_audio(fpath)
            if audio is None or len(audio) < 1600:
                continue

            results = det.process_clip(audio, sr)
            if not results:
                continue

            det_frames = sum(1 for r in results if r.detected)
            det_rate = det_frames / len(results)

            if det_rate > clip_thresh:
                n_detected += 1

        rate = n_detected / n_clips if n_clips > 0 else 0.0
        print(f"{class_name:<25s} {n_clips:>6d} {n_detected:>9d} {rate:>7.1%}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate Stage-1 detector on DroneAudioDataset"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the dataset from GitHub before validating",
    )
    parser.add_argument(
        "--dataset-dir", default="DroneAudioDataset",
        help="Path to the dataset directory (default: DroneAudioDataset)",
    )
    parser.add_argument(
        "--clip-thresh", type=float, default=0.30,
        help="Fraction of frames detected to count a clip (default: 0.30)",
    )
    parser.add_argument(
        "--require-specificity", action="store_true",
        help="Enable drone-specificity gate (AM + jitter)",
    )
    args = parser.parse_args()

    if args.download:
        download_dataset(args.dataset_dir)

    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset not found at: {args.dataset_dir}")
        print("Use --download to fetch it, or --dataset-dir to point to it.")
        sys.exit(1)

    validate(
        args.dataset_dir,
        clip_thresh=args.clip_thresh,
        require_specificity=args.require_specificity,
    )


if __name__ == "__main__":
    main()
