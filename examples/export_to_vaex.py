""" 
Here we show how to export data to veax-compatible format using API and how to read it in.

"""
from timspy.df import TimsPyDF
from timspy.vaex import TimsVaex

path = "/path/to/your/data.d"

D = TimsPyDF(path)
D.to_vaex("/path/to/where/to/save/data.hdf")
# above you can pass all other paramaters that are used by
# create_dataset function in h5py, as shown here:
# https://docs.h5py.org/_/downloads/en/2.6.0/pdf/

# Saving can take a while depending on the chosen algorithm.

V = TimsVaex("/path/to/where/to/save/data.hdf",
             "/path/to/your/data.d/analysis.tdf")# need sqlite

print(V)
TimsVaex.df
# #            frame    intensity    inv_ion_mobility    mz                  retention_time     scan    tof
# 0            1        9            1.6011418333261218  1174.6557905901582  0.326492080509831  33      312260
# 1            1        9            1.5999999999999996  733.4809407066116   0.326492080509831  34      220720
# 2            1        9            1.5999999999999996  916.952388791982    0.326492080509831  34      261438
# 3            1        9            1.5977164032508784  152.3556513940331   0.326492080509831  36      33072
# 4            1        9            1.5977164032508784  827.3114212681397   0.326492080509831  36      242110
# ...          ...      ...          ...                 ...                 ...                ...     ...
# 404,183,872  11553    9            0.6097471205799778  1171.1371127745708  1243.91933656337   909     311606
# 404,183,873  11553    9            0.6086254290341188  677.899262990993    1243.91933656337   910     207399
# 404,183,874  11553    9            0.6075037601757217  1084.4095190266144  1243.91933656337   911     295164
# 404,183,875  11553    9            0.6030173116029844  262.0435011990699   1243.91933656337   915     82016
# 404,183,876  11553    9            0.6018957561715719  1129.4302518267864  1243.91933656337   916     303778


# Then you can plot TIC 
V.plot_TIC(recalibrated=False)



import vaex
import matplotlib.pyplot as plt

I_per_scan = V.df.groupby(by='scan',
                          agg={'SummedIntenisty': vaex.agg.sum('intensity')})

# Then you can take the output, which is small and RAM-friendly,
# and do whatever you want with it :)
I_per_scan = I_per_scan.to_pandas_df().sort_values('scan')

#      scan  SummedIntenisty
# 629    33          2966589
# 630    34          2822660
# 631    35          2670091
# 868    36          2542816
# 632    37          2445243
# ..    ...              ...
# 624   913           372112
# 625   914           359317
# 626   915           365467
# 627   916           347171
# 628   917           347208

# [885 rows x 2 columns]

