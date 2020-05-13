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

# this concatenates the frames/windows that you want from this generator:
it = D.iter_arrays(frames=slice(1000,1100))
print(next(it))
print(next(it))
# or you can iterate data.frames
it = D.iter_dfs(frames=slice(1000,1100))
print(next(it))
print(next(it))

# to get some scans:
# let's find the most populated scan in the most intense frame
frame_no = D.frames.SummedIntensities.values.argmax()
F = D.df(frames=frame_no)
scan_no = F.groupby('scan').frame.count().values.argmax()
S = F.query('scan == @scan_no')
print(S)

SU = D.scan_usage()
binary=False
if binary:
    plt.axhline(y=self.max_scan, color='r', linestyle='-')
    plt.axhline(y=self.min_scan, color='r', linestyle='-')
SSU = np.count_nonzero(SU, axis=1) if binary else SU.sum(axis=1)
SSU_MS1 = SSU.copy()
SSU_MS1[D.ms2frames()] = 0
SSU_MS2 = SSU.copy()
SSU_MS2[D.ms1frames()] = 0
f = D.frame_indices()
plt.vlines(f,0, SSU_MS1, colors='orange')
plt.show()
D[0]

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
