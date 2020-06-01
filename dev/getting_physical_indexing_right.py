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
# D = TimsDIA(p) # timsdata does not support :

X = D[100]
# X = D.phys[1000:1010]
X = D.biggest_frame

K = np.percentile(D.frames.NumPeaks, 50)
w = D.frames.index[D.frames.NumPeaks == K][0]
D.frames.NumPeaks[w]
X = D[w]

D.phys[0:10]

%%timeit
D.tof2mz(X.tof)

%%timeit
D.tof2mz_model(X.tof)

