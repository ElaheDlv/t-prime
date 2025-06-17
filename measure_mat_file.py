#!/usr/bin/env python3
"""
measure_waveform_length.py

Loads a MATLAB .mat file containing a 'waveform' variable and reports:
  - The original array shape
  - Total number of samples
  - Data type

Supports both v7 and v7.3 MAT files.

Usage:
  python measure_waveform_length.py /path/to/dataset.mat
"""
import argparse
from pathlib import Path
import scipy.io
import h5py
import numpy as np

def load_waveform_info(mat_path: Path):
    """Return tuple (shape, total_samples, dtype) for the 'waveform' variable."""
    try:
        data = scipy.io.loadmat(str(mat_path))
        # find 'waveform'
        if 'waveform' in data:
            arr = data['waveform']
        else:
            # fallback: first ndarray in file
            arr = next(v for v in data.values() if isinstance(v, (np.ndarray,)))
        shape = arr.shape
        total = int(np.prod(shape))
        dtype = arr.dtype
    except Exception:
        # likely a v7.3 MAT file
        with h5py.File(str(mat_path), 'r') as f:
            key = next(iter(f.keys()))
            ds = f[key]
            shape = tuple(ds.shape)
            total = int(np.prod(shape))
            dtype = ds.dtype
    return shape, total, dtype

def main():
    parser = argparse.ArgumentParser(
        description="Measure length and type of 'waveform' in a .mat file."
    )
    parser.add_argument(
        'mat_file',
        type=str,
        help='Path to the MATLAB .mat file'
    )
    args = parser.parse_args()

    mat_path = Path(args.mat_file)
    if not mat_path.exists():
        print(f"Error: File not found: {mat_path}")
        return

    shape, total, dtype = load_waveform_info(mat_path)
    print(f"File: {mat_path}\n"
          f"  waveform shape       : {shape}\n"
          f"  total samples        : {total}\n"
          f"  data type            : {dtype}")

if __name__ == '__main__':
    main()
