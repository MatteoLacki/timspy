import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from datetime import datetime

from timspy.df import TimsPyDF
import tqdm

DEBUG = False

ap = argparse.ArgumentParser(description='Calculate the Total Ion Current for a given selection of points.',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

default_condition = "inv_ion_mobility < .0009*mz + .4744 & \
                     inv_ion_mobility > .6 & \
                     inv_ion_mobility < 1.5"

ARG = ap.add_argument

ARG('folders',
    nargs='+',
    help="Path(s) to a timsTOF .d folder(s) containing 'analysis.tdf' and 'analysis.tdf_raw'.",
    type=pathlib.Path)

ARG('--output',
    help="Path where to save output to.",
    default=pathlib.Path(f'C:/TICS/TIC_{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}.csv'),
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

ARG("--plot",
    help="Plot the selection results.",
    action='store_true')

ARG("--verbose",
    help="Show CLI messages.",
    action='store_true')


args = ap.parse_args()

if DEBUG:
  print(args.__dict__)

assert all(folder.exists() for folder in args.folders), "The data folder unavailable."

all_TICs = []
all_selected_region_TICs = []

for folder in args.folders:

    D = TimsPyDF(folder) # get data handle

    # getting m/z range.
    min_mz = int(D.min_mz) - 1
    max_mz = int(D.max_mz) + 1
    mz_bin_borders = \
        np.linspace(min_mz, 
                    max_mz,
                    args.mz_bins_cnt+1)

    inv_ion_mobility_bin_borders = \
        np.linspace(D.min_inv_ion_mobility,
                    D.max_inv_ion_mobility,
                    args.inv_ion_mobility_bins_cnt+1)

    # getting intensities
    intensities_matrix, _, _ = \
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
    TIC = D.frames.SummedIntensities.sum()
    TIC_selected_region = int(total_intensity_within_selected_region)
    if args.verbose:
        print(folder)
        print('TIC:')
        print(TIC)
        print('TIC of selected region:')
        print(TIC_selected_region)
    
    all_TICs.append(TIC)
    all_selected_region_TICs.append(TIC_selected_region)

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

out = pd.DataFrame({ 'folder': [f.name for f in args.folders],
                     'path':   args.folders,
                     'TIC':    all_TICs,
                     'TIC_selected_region': all_selected_region_TICs})

args.output.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(path_or_buf=args.output, index=False)


