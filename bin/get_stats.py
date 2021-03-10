import argparse
import numpy as np
import pandas as pd
import pathlib

from timspy.df import TimsPyDF

ap = argparse.ArgumentParser(description='Calculate the Total Ion Current or peak counts for a given selection of points.',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ARG = ap.add_argument

ARG('folders',
    nargs='+',
    help="Path(s) to a timsTOF .d folder(s) containing 'analysis.tdf' and 'analysis.tdf_raw'.",
    type=pathlib.Path)

ARG('--output',
    help="Path where to save output to.",
    default="",
    type=pathlib.Path)

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
# frames_stats_list = []
# borders_list = []

variables = ("mz","inv_ion_mobility") if args.statistic == "peak_count" else ("mz","inv_ion_mobility","intensity")

# for folder in args.folders:
#     if args.verbose:
#         print(f"Dealing with: {folder}")
#     D = TimsPyDF(folder)
#     frames_stats, borders = D.bin_frames(variables=variables,
#                                          min_row=50,
#                                          max_row=2000,
#                                          min_column=.3,
#                                          max_column=1.8,
#                                          bins_row=args.mz_bins_cnt,
#                                          bins_column=args.inv_ion_mobility_bins_cnt,
#                                          desaggregate=False,
#                                          return_df=True, 
#                                          verbose=args.verbose)
#     # frames_stats_list.append(frames_stats)
#     # borders_list.append(borders)

#     stats_long = pd.melt(frames_stats,
#                          ignore_index=False,
#                          value_name=args.statistic).reset_index()
#     statistic_total = stats_long[args.statistic].sum()
#     multiply_charged_statistic_total = stats_long.query(args.condition)[args.statistic].sum()
#     statistic_totals.append(statistic_total)
#     multiply_charged_statistic_totals.append(multiply_charged_statistic_total)

# out = pd.DataFrame({ "folder": [f.name for f in args.folders],
#                      "path":   args.folders,
#                      args.statistic: statistic_totals,
#                      f"{args.statistic}_multiply_charged": multiply_charged_statistic_totals})

if str(args.output) == ".":
    output = f"./{args.statistic}.csv"
else:
    output = args.output

print(output)
# out.to_csv(path_or_buf=args.output, index=False)
