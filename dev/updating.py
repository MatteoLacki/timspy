%load_ext autoreload
%autoreload 2
from timspy.timspydf import TimsPyDF
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 15)
import matplotlib.pyplot as plt

path = "/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d"
D = TimsPyDF(path)

D.plot_TIC(recalibrated=True)
D.plot_TIC(recalibrated=False)

# D.plot_intensity_given_mz_dt(imshow_kwds={'aspect':'auto',
# 										  'cmap': plt.get_cmap('inferno')})

# D.plot_intensity_given_mz_dt(imshow_kwds={'aspect':'auto',
# 										  'cmap': plt.get_cmap('inferno')},
# 							 mz_bin_borders=np.linspace(500, 2500, 10001),
#                              dt_bin_borders=np.linspace(0.8, 1.7, 201),
#                              intensity_tranformation=np.exp)

# I, mz_bin_borders, dt_bin_borders = D.intensity_given_mz_dt(mz_bin_borders=np.linspace(500, 2500, 2001),
#                         dt_bin_borders=np.linspace(0.8, 1.7, 201))

# plt.imshow(np.sqrt(I),
# 		   interpolation='lanczos',
# 		   aspect='auto',
#            extent=[mz_bin_borders[0],
#            		   mz_bin_borders[-1],
#            		   dt_bin_borders[0],
#            		   dt_bin_borders[-1]],
#            cmap=plt.get_cmap('inferno'))
# plt.show()


