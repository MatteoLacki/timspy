%load_ext autoreload
%autoreload 2
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 15)
import numpy as np
import pathlib
from pprint import pprint
import matplotlib.pyplot as plt
import scikit

from timspy.df import TimsPyDF
from timspy.dia import TimsPyDIA



# path = pathlib.Path('path_to_your_data.d')
path = pathlib.Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
D = TimsPyDF(path) # get data handle

# naive estimator: the to
D.frames.SummedIntensities
D.summary()
D.maxscan()


window_cnt_mz = 100
window_cnt_scan = 100

min_scan = 0
max_scan = D.frames.NumScans.max()
# another thing: how does this compare to 
GlobalMetadata = D.table2df('GlobalMetadata').set_index('Key')
min_mz = int(float(GlobalMetadata.Value['MzAcqRangeLower'])) - 1
max_mz = int(float(GlobalMetadata.Value['MzAcqRangeUpper'])) + 1
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

# return I, mz_bin_borders, inv_ion_mobility_bin_borders


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





