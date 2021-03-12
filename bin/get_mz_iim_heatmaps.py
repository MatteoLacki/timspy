import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import pathlib
import tqdm
from fast_histogram import histogram2d

from timspy.df import TimsPyDF

ap = argparse.ArgumentParser(description='Calculate a selection of stats for each raw folder.',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ARG = ap.add_argument

ARG('folders',
    nargs='+',
    help="Path(s) to a timsTOF .d folder(s) containing 'analysis.tdf' and 'analysis.tdf_raw'.",
    type=pathlib.Path)

ARG('--output_folder',
    help="Path to output folder to save output to.",
    default=".",
    type=pathlib.Path)

ARG('--mz_bins_cnt', 
    default=1000,
    type=int,
    help='Number of bins to divide the m/z axis into.')

ARG('--inv_ion_mobility_bins_cnt', 
    default=100,
    type=int,
    help='Number of bins to divide the inverse ion mobility axis into.')

ARG("--max_power",
    default=20,
    type=int,
    help="The power in soft maximum.")

ARG("--verbose",
    help="Show CLI messages.",
    action='store_true')


args = ap.parse_args()

# class MockArgs(object):
#     pass

# args = MockArgs()
# args.folders = list(pathlib.Path("/home/matteo/raw_data").glob("majestix/*.d"))
# args.folders = [pathlib.Path("/mnt/ms/majestix_rawdata/RAW/M201203_001_Slot1-1_1_696.d"),
#                 pathlib.Path("/mnt/ms/majestix_rawdata/RAW/M201203_002_Slot1-1_1_697.d")]
# args.output_folder = pathlib.Path("test_outputs")
# args.verbose = True
# args.mz_bins_cnt = 200
# args.inv_ion_mobility_bins_cnt = 100
# args.max_power = 20

analysis_time = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
variables = ("mz","inv_ion_mobility","intensity")

datasets = {folder: TimsPyDF(folder) for folder in args.folders}

assert len(datasets) >= 1, "No datasets selected!"
D = datasets[args.folders[0]]
min_mz, max_mz, min_iim, max_iim = D.min_mz, D.max_mz, D.min_inv_ion_mobility, D.max_inv_ion_mobility

for D in datasets.values():
    min_mz = min(D.min_mz, min_mz)
    max_mz = max(D.max_mz, max_mz)
    min_iim = min(D.min_inv_ion_mobility, min_iim)
    max_iim = max(D.max_inv_ion_mobility, max_iim)

bins = mz_bins, iim_bins = args.mz_bins_cnt, args.inv_ion_mobility_bins_cnt
bin_range = ((min_mz, max_mz), (min_iim, max_iim))

mz_borders = np.linspace(min_mz, max_mz, mz_bins+1)
iim_borders= np.linspace(min_iim, max_iim, iim_bins+1)

mz_mids = (mz_borders[1:] + mz_borders[:-1]) / 2.0
iim_mids = (iim_borders[1:] + iim_borders[:-1]) / 2.0

def save2csv(X, name, folder):
    X = pd.DataFrame(X, index=mz_mids, columns=iim_mids)
    X.to_csv(args.output_folder/f"{name}__{folder.stem}__{analysis_time}.csv")

# folder = args.date[0]
for folder in args.folders:
    if args.verbose:
        print(f"Dealing with: {folder}")

    D = datasets[folder]    
    frame_numbers = D.ms1_frames
    
    TIC_mean = np.zeros(shape=bins, dtype=float)
    TIC_p = np.zeros(shape=bins, dtype=float)
    TIC_sum = np.zeros(shape=bins, dtype=float)

    # frameNo = 1
    for frameNo in tqdm.tqdm(frame_numbers):
        frame = D.query(frameNo, variables)
        x = frame[variables[0]]
        y = frame[variables[1]]
        w = frame[variables[2]]
        peak_counts_frame = histogram2d(x, y, bins=bins, range=bin_range)
        TIC_frame = histogram2d(x, y, bins=bins, range=bin_range, weights=w)
        TIC_sum += TIC_frame
        TIC_mean_frame = np.zeros(shape=bins)
        TIC_mean_frame[peak_counts_frame > 0] = TIC_frame[peak_counts_frame > 0] / peak_counts_frame[peak_counts_frame > 0]
        TIC_mean += TIC_mean_frame
        TIC_p_frame = histogram2d(x, y, bins=bins, range=bin_range, weights=w**args.max_power)
        TIC_p += TIC_p_frame
    # finishing touches
    TIC_mean /= len(frame_numbers)
    TIC_p = TIC_p**1/args.max_power
    # storing
    save2csv(TIC_sum, "TIC_sum", folder)
    save2csv(TIC_mean,"TIC_mean",folder)
    save2csv(TIC_p,"TIC_maxes",folder)

