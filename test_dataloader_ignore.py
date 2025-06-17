import sys, os
sys.path.insert(0, os.getcwd())

from preprocessing.TPrime_dataset import TPrimeDataset
import os

#DATA_ROOT = '/home/trudes/Projects/t-prime/data/DATASET1_1'
DATA_ROOT = '/home/trudes/Projects/t-prime/data/DATASET_QYLUR_ANOM_small_frames'
ds = TPrimeDataset(
    protocols=['noAnomaly','cfoAnomaly','envDistortionAnomaly','pilotJamAnomaly','powerAnomaly'],
    #protocols=['802_11ax', '802_11b_upsampled', '802_11n', '802_11g'],
    slice_len=512,
    ds_type='train',
    slice_overlap_ratio=0.5,
    raw_data_ratio=1.0,
    ds_path=DATA_ROOT
)
print("Found folders:", os.listdir(ds.ds_path))
print("Total training samples:", len(ds))
print("Per-class raw file counts:", ds.n_sig_per_class)