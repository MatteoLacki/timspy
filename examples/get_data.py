"""How to get the data."""
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import vaex as vx

from timspy.timspy import TimsDIA
from timspy.plot import plot_spectrum
from timsdata import TimsData

# plt.style.use('dark_background')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)

# I have my data here
p = Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')

# p can be a simple 'string' too
D = TimsDIA(p)

# initiating D automatically gives you two data.frames:
print(D.windows)# info on available windows
# for completeness, window 0 corresponds to MS1.

# each window is a rectangle defined by the scans it contains and the extent in the m/z
print(D.windows.loc[1])
# you can plot windows
D.plot_windows()
D.peakCnts_massIdxs_intensities(100, 100, 101)
D.MS1_frameNumbers()


D.frames
D[1:10, 1:599]
D['rt > 10 and rt < 50', 1:599]
D['MsMsType == 0 and rt < 50', 1:599]

D.iter[1:10, 1:599]

class ComfyIter(object):
    def __init__(self, D):
        self.D = D

    def __getitem__(self, x):
        




D.global_TIC()


D.min_frame
D.max_frame
D.plot_peak_counts()
D[[1,2,10], 10:100]
D[[1,2,10], [10, 50]]

x = slice(2, 50, 3)
x = slice(2, 50)
range(x.start, x.stop, x.step)
for f in D.frames.query('MsMsType == 0').index.get_level_values('frame'):
    print(f)




np.r_[slice(2, 50, 3)]
np.r_[20]
np.r_[[20, 340], 1:45]

parse_slice(x)
parse_slice(10)


# and plot windows belonging to particular window groups
D.plot_windows('window_gr in (1,4)')

# another data frame contains info on the frames
print(D.frames)
# each frame here has a 'rt' value assigned: that's the retention time.

# to get data in one frame (say the 1000th):
F = D.df(frames=1000)
D[1000:1050]
print(F)

%%timeit
F = D.df(frames=11553)


# this is much more flexible:
F = D.df(frames=slice(1000,1500,3))
print(F)
F1 = D.df(frames=slice(1000,1500), window_grs=slice(1,10))
print(F1)
F2 = D.df(frames=slice(10,2300), filter_frames='rt >= 100 & rt <= 1000')
print(F2)

D[1000:1050:2]

# this will also work, but kill it: you will get out of RAM quickly
# D.df()
## The '.df' method is a wrapper around '.array' method:
A = D.array(frames=slice(1000,1100))

# if you want to slice ONLY with frames, you can do it this way too:
D[1000:1100]
D[[100,102,1042]]
D[10:540:10]
# scans are not implemented yet: I will need to get to C to make this efficient

D.frame_array(10, 100, 918)


# this concatenates the frames/windows that you want from this generator:
it = D.iter_arrays(frames=slice(1000,1100))
print(next(it))
print(next(it))
# or you can iterate data.frames
it = D.iter_dfs(frames=slice(1000,1100))
print(next(it))
print(next(it))

np.r_[1:10]
it = D.iter_arrays(10)
next(it)
D[10]
np.r_[10:23]


np.r_[np.r_[10]]
D.array(10)

list(D.iter_arrays(np.r_[10]))

D.array(frames=[10])

frames = 10

np.concatenate(list(D.iter_arrays(frames)), axis=0)

D.array(frames=[10,20])
frames = np.r_[[10,20], 12]
frames.sort()




# to get some scans:
# let's find the most populated scan in the most intense frame
frame_no = D.frames.SummedIntensities.values.argmax()
F = D.df(frames=frame_no)
scan_no = F.groupby('scan').frame.count().values.argmax()
S = F.query('scan == @scan_no')
print(S)

D.plot_scan_usage()
D.plot_overview()

# to change mz_idx to m/z there are two possibilities.
# 0. use the built in method
MZ = D.mzIdx2mz(S.mz_idx)
I = S.i
plot_spectrum(MZ, I)

# or used a fitted model
MS2 = D.mzIdx2mz_model(S.mz_idx)
D.plot_models()

# making it all faster
hdf5_files = [str(f) for f in p.glob('raw/*.hdf5')]
R = vx.open_many(hdf5_files)

R['rt'] = D.frame2rt_model(R.frame)
R['im'] = D.scan2im_model(R.scan)
R['mz'] = D.mzIdx2mz_model(R.mz_idx)

R.plot(R.mz, R.im, shape=(1000,919))
plt.show()
D.plot_models()

frames = range(1,100)
list(frames)


from collections import Counter
PC = D.peak_counts()
PC.max(axis=1).max()

next(D.iterScans(100, 500, 600))

x = D.get_peakCnts_massIdxs_intensities_array(1000, 0, 918)
x[0:918]
sum(x[0:918])
x[0:918]
D[10]




ss = D.readScans(1,0,918)
len(ss)


import pandas as pd


frames = range(D.min_frame, D.max_frame+1) if frames is None else frames
s = self.min_scan if min_scan is None else min_scan
S = self.max_scan if max_scan is None else max_scan
peaks = [self.get_peakCnts_massIdxs_intensities_array(int(f),s,S)[s:S]
         for f in frames]


class Scans(pd.DataFrame):
    def __init__(self, S, frames, min_scan, max_scan):
        super().__init__(S)
        self.columns = range(min_scan, max_scan)
        self.index = frames

    def plot(self, show=True, **plt_kwds):
        """Show number of peaks found in each scan in each frame, or the number of non-empty scans in each frame.

        binary (boolean): plot only scan usage.
        """
        import matplotlib.pyplot as plt
        plt.imshow(self.T, **plt_kwds)
        if show:
            plt.show()

    def plot1d(self, binary=False, color='orange', show=True, **vlines_kwds):
        import matplotlib.pyplot as plt

        if binary:
            plt.axhline(y=min(self.columns), color='r', linestyle='-')
            plt.axhline(y=max(self.columns), color='r', linestyle='-')
        S = np.count_nonzero(self, axis=1) if binary else self.sum(axis=1)
        if 'color' not in vlines_kwds:
            vlines_kwds['color'] = color
        plt.vlines(S.index,0, S, **vlines_kwds)
        if show:
            plt.show()



D.windows

SU = D.scan_usage()
SU.plot()
SU.plot1d()
plt.axhline(y=D.max_scan, color='r', linestyle='-')
plt.axhline(y=D.min_scan, color='r', linestyle='-')
SSU = np.count_nonzero(SU, axis=1)
SSU_MS1 = SSU.copy()

len(SSU_MS1)
SSU_MS1[D.ms2frames()-1] = 0
SSU_MS2 = SSU.copy()
SSU_MS2[D.ms1frames()-1] = 0
f = D.frame_indices()
plt.vlines(f,0, SSU_MS1, colors='orange', label='MS1')
plt.plot(f, SSU_MS2, c='grey', label='MS2')
plt.legend()
plt.show()

# we should make a consistant use of frames here

