%load_ext autoreload
%autoreload 2
from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 500)

from timspy.timspy import TimsDIA, TimsPyDF

p = Path('/mnt/samsung/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
D = TimsDIA(p)
D.framesq
E = TimsPyDF(p)
E.frames



E[11553]

# initiating D automatically gives you two data.frames:
print(D.windows)# info on available windows
# for completeness, window 0 corresponds to MS1.

# each window is a rectangle defined by the scans it contains and the extent in the m/z
print(D.windows.loc[1])
# you can plot windows
D.plot_windows()
D.peakCnts_massIdxs_intensities(100, 100, 101)
D.MS1_frameNumbers()

