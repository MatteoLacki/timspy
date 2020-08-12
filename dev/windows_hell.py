"""How to get the data."""
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from timspy.timspy import TimsDIA, TimsPyDF
from timspy.plot import plot_spectrum
from timsdata import TimsData

# plt.style.use('dark_background')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)

# I have my data here
p = Path(r'X:\external\2019_2020_Bruker_Kooperation\BrukerMIDIA\data_sept_2019\MIDIA_CE10_precursor\20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_25ms_Slot1-9_1_495.d')


p = Path(r'X:\external\2019_2020_Bruker_Kooperation\Bruker_MHCs\RawFiles\200122_AUR_2col90min_TenzerMHC_1_1_Slot1-3_1_1695.d')

# T = TimsDIA(p) # timsdata does not support :
T = TimsPyDF(p) # timsdata does not support :
T.phys[0:10]


