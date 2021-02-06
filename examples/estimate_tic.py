%load_ext autoreload
%autoreload 2
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from itertools import product

from timspy.df import TimsPyDF

# path = pathlib.Path('path_to_your_data.d')
path = pathlib.Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
D = TimsPyDF(path) # get data handle
D.tables_names()

# naive estimator: the to
D.frames.SummedIntensities

# Cutting away the cloud with ones:


mz_bins_cnt = 1000
inv_ion_mobility_bins_cnt = 100

# another thing: how does this compare to 
min_mz = int(D.min_mz) - 1
max_mz = int(D.max_mz) + 1
mz_bin_borders = np.linspace(min_mz, max_mz, mz_bins_cnt)

inv_ion_mobility_bin_borders = np.linspace(D.min_inv_ion_mobility,
                                           D.max_inv_ion_mobility,
                                           inv_ion_mobility_bins_cnt+1)

# getting initial intenisities
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
cond = "inv_ion_mobility < .0009*mz + .4744 & \
        inv_ion_mobility > .6 & \
        inv_ion_mobility < 1.5"
plot_selection = True

# the real code!
iim_bin_mids = (inv_ion_mobility_bin_borders[1:] + inv_ion_mobility_bin_borders[:-1]) / 2
mz_bin_mids = (mz_bin_borders[1:] + mz_bin_borders[:-1]) / 2

total_intensity = 0.0
if plot_selection:
    Xs = []
for j,iim in enumerate(iim_bin_mids):
    X = pd.DataFrame({"inv_ion_mobility":iim,
                  "mz":mz_bin_mids,
                  "intensity":intensities_matrix[:,j]}).query(cond)
    if plot_selection:
        Xs.append(X)
    total_intensity += X.intensity.sum()


if plot_selection:
    XX = pd.concat(Xs)
    plt.scatter(XX.mz, XX.inv_ion_mobility, s=.1, c='green')
    D.plot_intensity_given_mz_inv_ion_mobility(
        intensities_matrix, 
        mz_bin_borders,
        inv_ion_mobility_bin_borders,
        intensity_transformation = lambda x:x,
        show=False)

    plt.show()




