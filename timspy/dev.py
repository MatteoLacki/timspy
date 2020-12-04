%load_ext autoreload
%autoreload 2
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5)
import matplotlib.pyplot as plt
import pathlib
from collections import Counter
import numpy as np
import vaex

from timspy.dia import TimsPyDIA
from timspy.vaex import TimsVaex

path = pathlib.Path("/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d")
D = TimsPyDIA(path)
D.frames_meta()



plt.plot(D.frames.MsMsType)
plt.show()
D.frames[D.frames.MsMsType != 0]
D.windows
len(Ids)
len(D.windows)


Counter(D.windows.WindowGroup)



hdf_path = "/home/matteo/Projects/bruker/hdf5data/test_defaults.hdf5"
tdf_path = path/"analysis.tdf"

V = TimsVaex(hdf_path, tdf_path)

V.df.groupby()
V.df.filter()
frames = V.ms1_frames

X = V.df
Y = X[X.frame.isin(frames)]

Y.sum(Y)

intensity_transformation = np.sqrt

# s, S = V.minmax('scan')
# f, F = V.minmax('frame')
I = Y.sum(Y.intensity,
          binby=[Y.scan, Y.frame],
          limits=[[0,1000], [f-.5, F+.5]],
          shape=[100, 1000])

# mz_bin_borders = np.arange(s,S)
# dt_bin_borders = np.arange(f,F) 
plt.imshow(np.sqrt(I),
           # extent=[mz_bin_borders[0], mz_bin_borders[-1],
           #         dt_bin_borders[0], dt_bin_borders[-1]],
           interpolation='lanczos',
           aspect='auto',
           cmap=plt.get_cmap('inferno'))
plt.xlabel("Mass / Charge")
plt.ylabel("Drift Time")
plt.title("Total Intensity")
plt.show()



# D.plot_TIC(recalibrated=True)
# D.plot_TIC(recalibrated=False)
# D.plot_intensity_given_mz_dt(imshow_kwds={'aspect':'auto',
#                                         'cmap': plt.get_cmap('inferno')})

# D.plot_intensity_given_mz_dt(imshow_kwds={'aspect':'auto',
#                                         'cmap': plt.get_cmap('inferno')},
#                            mz_bin_borders=np.linspace(500, 2500, 10001),
#                              dt_bin_borders=np.linspace(0.8, 1.7, 201),
#                             intensity_tranformation=np.exp)

# I, mz_bin_borders, dt_bin_borders = D.intensity_given_mz_dt(mz_bin_borders=np.linspace(500, 2500, 2001),
#                         dt_bin_borders=np.linspace(0.8, 1.7, 201))

# plt.imshow(np.sqrt(I),
#          interpolation='lanczos',
#          aspect='auto',
#            extent=[mz_bin_borders[0],
#                      mz_bin_borders[-1],
#                      dt_bin_borders[0],
#                      dt_bin_borders[-1]],
#            cmap=plt.get_cmap('inferno'))