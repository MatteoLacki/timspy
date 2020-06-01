"""How to get the data."""
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from timspy.timspy import TimsDIA, TimsPyDF
from timspy.plot import plot_spectrum


# plt.style.use('dark_background')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)

# I have my data here
p = Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
TD = TimsData(p) # timsdata does not support :
D = TimsPyDF(p) # timsdata does not support :
D = TimsDIA(p) # timsdata does not support :
D.plot_models()
X = D[100]
# X = D.phys[1000:1010]
X = D.biggest_frame

K = np.percentile(D.frames.NumPeaks, 50)
w = D.frames.index[D.frames.NumPeaks == K][0]
D.frames.NumPeaks[w]
X = D[w]

D.phys[0:10]
X = D[[1000, 2300, 5000]]
print(X)
print(X)
%%timeit
D.tof2mz(X.tof)

%%timeit
D.tof2mz_model(X.tof)

it = D.iter[:10000]
next(it)

D.phys[1:10, 0.7:1.5]
it = D.physIter[1:10]
print(next(it))

D.plot_overview(1000, 4000)
D.plot_peak_counts()

D.windows

D.plot_windows("window_gr in [1,5,10]")
next(D.iter['window_gr == 1'])