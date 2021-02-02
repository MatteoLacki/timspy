import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

from timspy.df import TimsPyDF
import tqdm

DEBUG = False

ap = argparse.ArgumentParser(description='Calculate the Total Ion Current for a given selection of points.',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

default_condition = "inv_ion_mobility < .0009*mz + .4744 & \
                     inv_ion_mobility > .6 & \
                     inv_ion_mobility < 1.5"

ARG = ap.add_argument

ARG('folder',
    help="Path to a timsTOF .d folder with data.",
    type=pathlib.Path)

ARG('--condition', 
    default=default_condition,
    help='A string that ties mass over charge ratios and inverse ion mobilities.')

ARG('--mz_bins_cnt', 
    default=1000,
    type=int,
    help='Number of bins to divide the m/z axis into.')

ARG('--inv_ion_mobility_bins_cnt', 
    default=100,
    type=int,
    help='Number of bins to divide the inverse ion mobility axis into.')

ARG('--min_binning_inv_ion_mobility', 
    default=0.0,
    type=float,
    help='Minimal inverse ion mobility for binning.')

ARG('--max_binning_inv_ion_mobility', 
    default=2.0,
    type=float,
    help='Maximal inverse ion mobility for binning.')

ARG("--plot",
    help="Plot the selection results.",
    action='store_true')

args = ap.parse_args()

if DEBUG:
  print(args.__dict__)

assert args.folder.exists(), "The data folder unavailable."

D = TimsPyDF(args.folder) # get data handle


# getting m/z range.
GlobalMetadata = D.table2df('GlobalMetadata').set_index('Key')
min_mz = int(float(GlobalMetadata.Value['MzAcqRangeLower'])) - 1
max_mz = int(float(GlobalMetadata.Value['MzAcqRangeUpper'])) + 1
mz_bin_borders = np.linspace(min_mz, max_mz, args.mz_bins_cnt+1)

inv_ion_mobility_bin_borders = np.linspace(args.min_binning_inv_ion_mobility,
                                           args.max_binning_inv_ion_mobility,
                                           args.inv_ion_mobility_bins_cnt+1)

# getting intensities
intensities_matrix, mz_bin_borders, inv_ion_mobility_bin_borders = \
    D.intensity_given_mz_inv_ion_mobility(D.ms1_frames,
                                          mz_bin_borders,
                                          inv_ion_mobility_bin_borders,
                                          verbose=True)

def abline(slope, intercept, **kwds):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', **kwds)


# # getting defaults
# x0,y0 = 331.5, 0.767 
# x1,y1 = 1254.0, 1.581
# A = (y1-y0)/(x1-x0)
# B = y0 - A*x0

# D.plot_intensity_given_mz_inv_ion_mobility(
#     intensities_matrix, 
#     mz_bin_borders,
#     inv_ion_mobility_bin_borders,
#     intensity_transformation = lambda x:x,
#     show=False)
# abline(A, B)
# abline(.0009, .4744, c='red')
# abline(.001, .474, c='green')
# plt.show()
#  hence: default condition:
# cond = "inv_ion_mobility < .0009*mz + .4744 & \
#         inv_ion_mobility > .6 & \
#         inv_ion_mobility < 1.5"

iim_bin_mids = (inv_ion_mobility_bin_borders[1:] + inv_ion_mobility_bin_borders[:-1]) / 2
mz_bin_mids = (mz_bin_borders[1:] + mz_bin_borders[:-1]) / 2

total_intensity_within_selected_region = 0.0
if args.plot:
    Xs = []
for j,iim in enumerate(iim_bin_mids):
    X = pd.DataFrame({"inv_ion_mobility":iim,
                      "mz":mz_bin_mids,
                      "intensity":intensities_matrix[:,j]}).query(args.condition)
    # general condition work, see pandas entry on the query method
    if args.plot:
        Xs.append(X)
    total_intensity_within_selected_region += X.intensity.sum()

# naive estimator: the to
print('TIC:')
print(D.frames.SummedIntensities.sum())
print('TIC of selected region:')
print(total_intensity_within_selected_region)


if args.plot:
    XX = pd.concat(Xs)
    plt.scatter(XX.mz, XX.inv_ion_mobility, s=.1, c='green')
    D.plot_intensity_given_mz_inv_ion_mobility(
        intensities_matrix, 
        mz_bin_borders,
        inv_ion_mobility_bin_borders,
        intensity_transformation = lambda x:x,
        show=False)
    plt.show()




