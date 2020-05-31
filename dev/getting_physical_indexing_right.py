"""How to get the data."""
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from timspy.timspy import TimsDIA
from timspy.plot import plot_spectrum
from timsdata import TimsData

# plt.style.use('dark_background')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)

# I have my data here
p = Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
# TD = TimsData(p) # timsdata does not support :
D = TimsDIA(p) # timsdata does not support :
X = D[1:100]


X['rt'] = D.frame2rt_model(X.frame)
X['im'] = D.scan2im_model(X.scan)
X['tof'] = D.tof2mz_model(X.tof)
