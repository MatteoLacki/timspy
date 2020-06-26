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
p = Path(r'X:\external\2019_2020_Bruker_Kooperation\BrukerMIDIA\data_sept_2019\MIDIA_CE10_precursor\20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_25ms_Slot1-9_1_495.d')
# TD = TimsData(p) # timsdata does not support :
TD = TimsDIA(p) # timsdata does not support :

update_wrapper(TD.iter, TD.iter_arrays)

TD.iter.__getitem__
?TD.iter_arrays


TD[1:10,:]
next(TD.iter[1:10, 100:500])
list(TD.iter[1:10, 100:500])
TD[1:10, 100:500]
TD[1:100, 100]
list(TD.iter[1:100, 100])


TD[1:10, 100:500].dtype
TD[[10, 20, 30], 100:500]
TD[[10, 20, 30], [40, 49]]
TD[[10, 20, 30], [41, 60]]
TD[:20, [41, 60]]
TD[11552,10]
TD[11552:,10] # exception will be raised automatically!

TD[(i**2 for i in range(1,10)), 10:50]
