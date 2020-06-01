"""How to get the data."""
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from timspy.timspy import TimsDIA, TimspyDF
from timspy.array_ops import which_min_geq, which_max_leq
from timspy.plot import plot_spectrum
from timsdata import TimsData

# plt.style.use('dark_background')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)

# I have my data here
p = Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
# TD = TimsData(p) # timsdata does not support :
D = TimspyDF(p) # timsdata does not support :
# D = TimsDIA(p) # timsdata does not support :

X = D[100]
X['rt'] = D.frame2rt_model(X.frame)
X['im'] = D.scan2im_model(X.scan)
X['mz'] = D.tof2mz_model(X.tof)

x = D.scan2im_model.x
y = D.scan2im_model.y

it = D.iter_physical(slice(1,6), append=True)
next(it)
D.frame2rt([6,7,8,9])
D.frames.rt


it = D.iter_physical((slice(1243.91,None), slice(None)), append=False)
next(it)

D.rt2frame([1243.91])
D.rt2frame([1243.92])

D.phys[1243.91:]

D.min_frame
D.max_frame
D.frames.rt

D.max_scan
D.frames.rt[D.frames.rt>.9]
D.frames.rt[D.frames.rt>5.797]
D.phys[1:6]
D.rt2infFrame(6)
D.max_frame

D[53]
D.frame2rt(52)
D.frames.rt[52]
D.frames.rt[52]
D.rt2infFrame([6])
D[53]
D.frames.rt[51:55]

next(D.iter_physical(slice(5,6), append=True))
D.frames.rt[D.frames.rt>5]

D.frame2rt_model([47])
D.frame2rt([42, 43, 44, 45])
D.frames.rt[[42, 43, 44, 45]]
D.frames.rt[[42, 43, 44, 45]]

rt = D.frames.rt
frames = D.frames.index
rts = D.frames.rt.values

frames[which_min_geq(rts, [0.35, .8])].values
D.rt2supFrame([1,2])
D.rt2infFrame([10])
D.rt2infFrame([D.frames.rt[88:92][91]])


D.frame2rt(X[:,0])
X.dtypes

D.scan2im_model((D.min_scan, D.max_scan))
D.rt2frame(11135)

D.scan2im_model.naive_inv(im)

pd.RangeIndex(0, 100)
Z = pd.DataFrame(index=np.arange(10), columns=('rt', 'im', 'mz', 'i'), dtype=np.float64)
Z.rt = 
D.tof2mz_model