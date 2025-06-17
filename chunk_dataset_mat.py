#!/usr/bin/env python3
"""
chunk_to_frame_len.py

Splits each anomaly .mat file into segments whose length matches an example
original dataset waveform frame. This ensures pre-chunked files have the exact
same dimensionality as the protocol dataset, so TPrimeDataset can ingest them
without any generate_windows adjustments.

Usage:
  python chunk_to_frame_len.py \
    --input_dir /path/to/DATASET_QYLUR_ANOM_small \
    --orig_mat /path/to/DATASET1_1/802_11ax/802.11ax_IQ_frame_1.mat \
    [--output_dir /path/to/output_frames]
"""
import argparse
from pathlib import Path
import numpy as np
import scipy.io
import shutil
import h5py
import sys

def load_waveform(mat_path: Path) -> np.ndarray:
    """Load waveform array (flattened) from a .mat, v7 or v7.3."""
    try:
        data = scipy.io.loadmat(str(mat_path))
        if 'waveform' in data:
            arr = data['waveform']
        else:
            arr = next(v for v in data.values() if isinstance(v, np.ndarray))
        return arr.flatten()
    except Exception:
        with h5py.File(str(mat_path), 'r') as f:
            key = next(iter(f.keys()))
            return np.array(f[key]).flatten()

def measure_frame_len(orig_mat: Path) -> int:
    """Return the number of samples in the 'waveform' of an original dataset file"""
    wf = load_waveform(orig_mat)
    return wf.shape[0]

def chunk_file_to_frames(mat_path: Path,
                         meta_path: Path,
                         out_dir: Path,
                         frame_len: int):
    """Split a single anomaly file into full-length frames."""
    wf = load_waveform(mat_path)
    n_frames = len(wf) // frame_len
    stem = mat_path.stem
    for i in range(n_frames):
        start = i * frame_len
        seg = wf[start:start+frame_len]
        # make it a column vector
        seg_col = seg.reshape(-1,1)
        out_mat = out_dir / f"{stem}_frame{i:04d}.mat"
        scipy.io.savemat(str(out_mat), {'waveform': seg_col})
        # copy metadata per frame
        out_meta = out_dir / f"{stem}_frame{i:04d}{meta_path.suffix}"
        shutil.copy(str(meta_path), str(out_meta))
    print(f"  -> {stem}: {n_frames} frames of length {frame_len}")

def main():
    parser = argparse.ArgumentParser(
        description="Chunk anomaly signals into original-frame-sized MAT files"
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        help="Root with one subfolder per anomaly class"
    )
    parser.add_argument(
        '--orig_mat',
        required=True,
        help="One example original dataset .mat to measure frame length"
    )
    parser.add_argument(
        '--output_dir',
        required=False,
        help="Where to write frame-chunked data (defaults to <input_dir>_frames)"
    )
    args = parser.parse_args()

    in_root = Path(args.input_dir).expanduser().resolve()
    if not in_root.is_dir():
        sys.exit(f"Error: input_dir {in_root} not found")

    orig_mat = Path(args.orig_mat).expanduser().resolve()
    if not orig_mat.exists():
        sys.exit(f"Error: orig_mat {orig_mat} not found")

    frame_len = measure_frame_len(orig_mat)
    print(f"Measured original frame length = {frame_len}")

    out_root = Path(args.output_dir).expanduser().resolve() 
    if args.output_dir is None:
        out_root = in_root.with_name(in_root.name + '_frames')
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing frames to: {out_root}")

    # Process each class
    for cls in sorted(in_root.iterdir()):
        if not cls.is_dir():
            continue
        print(f"Class: {cls.name}")
        dest = out_root / cls.name
        dest.mkdir(exist_ok=True)
        mats = sorted(cls.glob('*.mat'))
        for matf in mats:
            # find metadata
            meta_yaml = matf.with_suffix('.yaml')
            meta_yml  = matf.with_suffix('.yml')
            if meta_yaml.exists(): metadata = meta_yaml
            elif meta_yml.exists(): metadata = meta_yml
            else:
                print(f"  [WARN] no metadata for {matf.name}, skipping")
                continue
            chunk_file_to_frames(matf, metadata, dest, frame_len)

    print("Done.")

if __name__ == '__main__':
    main()
