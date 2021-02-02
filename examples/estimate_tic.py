%load_ext autoreload
%autoreload 2
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 15)
import numpy as np
import pathlib
from pprint import pprint
import matplotlib.pyplot as plt
from skimage import feature
from skimage.feature import blob_dog, blob_log, blob_doh
from itertools import product

from timspy.df import TimsPyDF
from timspy.dia import TimsPyDIA



# path = pathlib.Path('path_to_your_data.d')
path = pathlib.Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
D = TimsPyDF(path) # get data handle

# naive estimator: the to
D.frames.SummedIntensities

# Cutting away the cloud with ones:
N = 1000
window_cnt_mz = N
window_cnt_scan = N

min_scan = 0
max_scan = D.frames.NumScans.max()
# another thing: how does this compare to 
GlobalMetadata = D.table2df('GlobalMetadata').set_index('Key')
min_mz = int(float(GlobalMetadata.Value['MzAcqRangeLower'])) - 1
max_mz = int(float(GlobalMetadata.Value['MzAcqRangeUpper'])) + 1

def getI(D, min_mz, max_mz, window_cnt_mz, min_scan, max_scan, window_cnt_scan):
    mz_bin_borders = np.linspace(min_mz, max_mz, window_cnt_mz)
    scan_bin_borders = np.linspace(min_scan, max_scan, window_cnt_scan)

    I = np.zeros(shape=(len(mz_bin_borders)-1,
                        len(scan_bin_borders)-1),
                 dtype=float)
    # float because numpy does not have histogram2d with ints 
    for X in D.query_iter(frames=D.ms1_frames,
                          columns=('mz','scan','intensity')):
        I_fr, _,_ = np.histogram2d(X.mz, X.scan,
                                   bins=[mz_bin_borders,
                                         scan_bin_borders], 
                                   weights=X.intensity)
        I += I_fr
    return I, mz_bin_borders, scan_bin_borders

I1000,_,_ = getI(D, min_mz, max_mz, 1000, min_scan, max_scan, 1000)
I50,  _,_ = getI(D, min_mz, max_mz, 50, min_scan, max_scan, 50)


plt.imshow(I.T,
           extent=[mz_bin_borders[0],
                   mz_bin_borders[-1],
                   scan_bin_borders[0],
                   scan_bin_borders[-1]],
           interpolation='none',
           aspect='auto',
           cmap='inferno',
           origin='lower')
plt.xlabel("Mass / Charge")
plt.ylabel("Scan Number")
plt.show()

intensity = I.flatten()
lg_intensity = np.log2(intensity[intensity>0])

plt.hist(lg_intensity, bins=100)
plt.hist(intensity[intensity>0], bins=1000)
plt.show()

# how to decipher which cloud is which?

intensities_matrix, mz_bin_borders, inv_ion_mobility_bin_borders = \
    D.intensity_given_mz_inv_ion_mobility()

D.plot_intensity_given_mz_inv_ion_mobility(
    intensities_matrix, 
    mz_bin_borders,
    inv_ion_mobility_bin_borders,
    intensity_transformation = lambda x: x
)


# # Line fitting again
# I.shape

Inorm = I / I.sum()
# # 99 = i
# # 49 = j

# I.dot(j)
I[0].dot(j)

i = np.arange(Inorm.shape[0])
j = np.arange(Inorm.shape[1])
Inorm_j = Inorm.dot(j)
i_Inorm = i.dot(Inorm)
sumI   = Inorm.sum()
sumIi  = i_Inorm.sum()
sumIj  = Inorm_j.sum()
sumIij = i_Inorm.dot(j)
sumIii = (i**2).dot(Inorm).sum()

# delta = sumI*sumIii - sumIi**2
# const = ( sumIii * sumIj - sumIi * sumIij) / delta
# slope = (-sumIj  * sumIi + sumI  * sumIij) / delta
# cyclic change in indices.
sumIjj = Inorm.dot(j**2).sum()
delta = sumI*sumIjj - sumIj**2
const = ( sumIjj * sumIi - sumIj * sumIij) / delta
slope = (-sumIi  * sumIj + sumI  * sumIij) / delta

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

plt.imshow(I,
           interpolation='none',
           aspect='auto',
           cmap='inferno',
           origin='lower')
abline(slope, const)
plt.show()
# fuck, doesn't work properly

def plot2d(I, show=True):
    plt.imshow(I,
               interpolation='none',
               aspect='auto',
               cmap='inferno',
               origin='lower')
    if show:
        plt.show()

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import img_as_float

im = I
image_max = ndi.maximum_filter(im, size=10, mode='constant')
# plot2d(image_max)

# ?peak_local_max
coordinates = peak_local_max(im, min_distance=10)
len(coordinates)

plot2d(I, show=False)
plt.scatter(coordinates[:,1], coordinates[:,0])
plt.show()


I = I1000
# diagonals = np.array([np.trace(I, -i) for i in range(-1000, 1001)])
# diagonals = np.array([np.diagonal(I, -i).max() for i in range(-N+2, N-2)])
diagonals = np.array([np.diagonal(I, -i).mean() for i in range(-I.shape[0]+2, I.shape[1]-2)])
plt.plot(diagonals)
plt.show()

I.shape
J = I
Need to put zeros

plot2d(I, show=True)
plot2d(I[:,::-1], show=True)
plot2d(np.tril(I[:,::-1], 200), show=True)

plot2d(np.tril(I[::-1,::-1], 40), show=True)



plt.plot(I.sum(axis=1));plt.show()
plt.plot(I.sum(axis=0));plt.show()

