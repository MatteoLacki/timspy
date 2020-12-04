# %load_ext autoreload
# %autoreload 2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# pd.set_option('display.max_rows', 20)
# pd.set_option('display.max_columns', 500)
# from pathlib import Path
# import vaex as vx

# from timspy.timspy import TimsDIA
# from timspy.iterators import ranges
# from time import time

# p = Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
# D = TimsDIA(p)
# # output_folder = Path("/home/matteo/Projects/bruker/data_dumps/prec_prec_100ms")
# output_folder = Path('/mnt/samsung/bruker/testHDF5/prec_prec_100ms')
# # D.to_hdf5(output_folder)

# D.tof2mz_model.plot()
# D.tof2mz_model.params

# df = vx.open_many([str(p) for p in output_folder.glob("*.hdf5")])
# df.plot(df.tof, df.scan, what=vx.stat.sum(df.i))
# plt.tight_layout()
# plt.show()


# df.plot(df.tof, df.scan, what=vx.stat.sum(df.i))
# plt.tight_layout()
# plt.show()


# # check if all stats are there
# x = df.count(df.i, binby=[df.scan], shape=1000)
# S = df.groupby(df.scan).agg({'i':'sum'})
# np.sort(S.scan.values)
# S = df.groupby(df.scan).agg({'i':'sum'})

