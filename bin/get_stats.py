import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import pathlib

from timspy.df import TimsPyDF
# from timspy.plot import heatmap

ap = argparse.ArgumentParser(description='Calculate the Total Ion Current or peak counts for a given selection of points.',
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

ARG('--save_plots',
    help="Save plots of the folders.",
    action='store_true')

ARG('--save_computed_data',
    help="Save raw data to the folder.",
    action='store_true')

ARG('--condition', 
    default="inv_ion_mobility < .0009*mz + .4744",
    help='A string that ties mass over charge ratios and inverse ion mobilities.')

ARG('--statistic', 
    default="TIC",
    choices=("TIC","peak_count"),
    help='The statistic to obtain.')

ARG('--mz_bins_cnt', 
    default=1000,
    type=int,
    help='Number of bins to divide the m/z axis into.')

ARG('--inv_ion_mobility_bins_cnt', 
    default=100,
    type=int,
    help='Number of bins to divide the inverse ion mobility axis into.')

ARG("--verbose",
    help="Show CLI messages.",
    action='store_true')

args = ap.parse_args()

statistic_totals = []
multiply_charged_statistic_totals = []

variables = ("mz","inv_ion_mobility") if args.statistic == "peak_count" else ("mz","inv_ion_mobility","intensity")

analysis_time = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')

for folder in args.folders:
    if args.verbose:
        print(f"Dealing with: {folder}")
    D = TimsPyDF(folder)
    frames_stats, borders = D.bin_frames(variables=variables,
                                         bins_row=args.mz_bins_cnt,
                                         bins_column=args.inv_ion_mobility_bins_cnt,
                                         desaggregate=False,
                                         return_df=True, 
                                         verbose=args.verbose)
    if args.save_computed_data:
        frames_stats.to_csv(path_or_buf=args.output_folder/f"{folder.stem}_{args.statistic}_{analysis_time}_all.csv")
    
    stats_long = pd.melt(frames_stats,
                         ignore_index=False,
                         value_name=args.statistic).reset_index()
    statistic_total = stats_long[args.statistic].sum()
    multiply_charged_statistic_total = stats_long.query(args.condition)[args.statistic].sum()
    statistic_totals.append(statistic_total)
    multiply_charged_statistic_totals.append(multiply_charged_statistic_total)

summary_df = pd.DataFrame({ "folder": [f.name for f in args.folders],
                     "path":   args.folders,
                     args.statistic: statistic_totals,
                     f"{args.statistic}_multiply_charged": multiply_charged_statistic_totals})

output_file = args.output_folder/f"{args.statistic}_{analysis_time}.csv"
summary_df.to_csv(path_or_buf=output_file, index=False)
