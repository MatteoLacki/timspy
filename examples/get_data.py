"""How to get the data.

UPDATE THIS!!!
"""
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
p.exists()
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

plt.style.use('ggplot')
plt.style.use('default')
D.plot_models(True, False, True)
plt.savefig('/home/matteo/Projects/bruker/models_vertical.pdf', 
    transparent=True,
    papertype='a1',
    dpi=1000)

?plt.savefig
# different indexing schemes.
D.frames
list(D.iter[1:5, 100:110])
list(D.iter[1:4])
list(D.iter['rt > 10 and rt < 20'])
D[1:5, 1:100]
D[1:100]
D[1:5, [33, 50]]
D[(i**2 for i in range(10)), [33, 50]]
D[[1,2,10], 10:100]
D[[1,2,10], [10, 50]]
D['rt > 10 and rt < 50', 1:599]
D['MsMsType == 0 and rt < 50', 1:599]
it = D.iter['MsMsType == 0 and rt < 50', 1:599]
next(it)
ms1it = D.iter_MS1()
next(ms1it)
next(ms1it)

D.plot_overview(1000,2000)
D.global_TIC()
D.plot_peak_counts()

X = D[1:100]

X['rt'] = D.frame2rt_model(X.frame)
X['im'] = D.scan2im_model(X.scan)
X['mz'] = D.mzIdx2mz_model(X.mz_idx)

X.physical[0.190156:]
X = X.query('rt >= 0.2 and rt <= 10')
X[['rt','im','mz']]

x = slice(2, 50, 3)
x = slice(2, 50)
range(x.start, x.stop, x.step)
for f in D.frames.query('MsMsType == 0').index.get_level_values('frame'):
    print(f)

ms1it = D.iter['MsMsType == 0']
next(ms1it)

D['MsMsType == 0 and rt < 10']

# and plot windows belonging to particular window groups
D.plot_windows('window_gr in (1,4)')

# another data frame contains info on the frames
print(D.frames)
# each frame here has a 'rt' value assigned: that's the retention time.


%%timeit
F = D[11553]
# 1.11 ms ± 28.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)  

D.frames.query('window_gr == 1')
D['window_gr == 1 and rt < 10']

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
it = D.iter[1000:1100]
print(next(it))
print(next(it))

# to get some scans:
# let's find the most populated scan in the most intense frame
frame_no = D.frames.SummedIntensities.values.argmax()
F = D[frame_no]
scan_no = F.groupby('scan').frame.count().values.argmax()
S = F.query('scan == @scan_no')
print(S)


# to change mz_idx to m/z there are two possibilities.
# 0. use the built in method
MZ = D.mzIdx2mz(S.mz_idx)
I = S.i
plot_spectrum(MZ, I)

# or used a fitted model
MS2 = D.mzIdx2mz_model(S.mz_idx)
D.plot_models()

# making it all faster
hdf5 = Path("/mnt/samsung/bruker/testHDF5/prec_prec_100ms")
hdf5_files = [str(f) for f in hdf5.glob('*.hdf5')]
p = Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
D = TimsDIA(p)
R = vx.open_many(hdf5_files)

R['rt'] = D.frame2rt_model(R.frame)
R['im'] = D.scan2im_model(R.scan)
R['mz'] = D.tof2mz_model(R.tof)
R.plot(R.mz, R.im, shape=(1000,919))
plt.show()




D.plot_models()

frames = range(1,100)
list(frames)



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

