"""How to get the data."""
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from timspy.timspy import TimsDIA, TimspyDF
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
D.max_non_empty_scan
D[100]
F = D.frames
F.index[F.MaxIntensity > 0][-1]

D.min_max_frames




X._info_repr()

from pandas._config import get_option
from pandas.io.formats import console
from io import StringIO


def repr(X):
    """
    Return a string representation for a particular DataFrame.
    """
    buf = StringIO("")
    max_rows = get_option("display.max_rows")
    min_rows = get_option("display.min_rows")
    max_cols = get_option("display.max_columns")
    max_colwidth = get_option("display.max_colwidth")
    show_dimensions = get_option("display.show_dimensions")
    if get_option("display.expand_frame_repr"):
        width, _ = console.get_console_size()
    else:
        width = None
    X.to_string(
        buf=buf,
        max_rows=max_rows,
        min_rows=min_rows,
        max_cols=max_cols,
        line_width=width,
        max_colwidth=max_colwidth,
        show_dimensions=show_dimensions,
    )
    return buf.getvalue()

repr(X)

X.head(3)
X.tail(3)

(len(X.__repr__().split('\n')) - 2) // 2 - 1

D.peak_count()
X = D[1:100]




D[D.min_frame]
next(D.iter[:,:])

it = D.iter_data_frames((slice(1,10),slice(None)))
next(it)
next(D.iter_data_frames(slice(None)))
D.iter[""]

list(D.iter[1:10,:])
list(D.iter[1:10,0:900])

for a in D.iter[1:10,0:900]:
    print(a)

next(D.iter_MS1())

X['rt'] = D.frame2rt_model(X.frame)
X['im'] = D.scan2im_model(X.scan)
X['mz'] = D.tof2mz_model(X.tof)


