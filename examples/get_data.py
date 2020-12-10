from pprint import pprint

from timspy.df import TimsPyDF


path = 'path/to/your/data.d'
D = TimsPyDF(path) # get data handle

print(D)
# TimsPyDF(404183877 peaks)

print(len(D)) # The number of peaks.
# 404183877 

D.intensity_per_frame() # Return combined intensity for each frame.
# array([ 95955, 579402, 907089, ..., 406508,   8097,   8633])

try:
    import opentims_bruker_bridge
    all_columns = ('frame','scan','tof','intensity','mz','inv_ion_mobility','retention_time')
except ModuleNotFoundError:
    print("Without Bruker proprietary code we cannot yet perform tof-mz and scan-dt transformations.")
    print("Download 'opentims_bruker_bridge' if you are on Linux or Windows.")
    print("Otherwise, you will be able to use only these columns:")
    all_columns = ('frame','scan','tof','intensity','retention_time')


# We consider the following columns:
print(all_columns)
# ('frame', 'scan', 'tof', 'intensity', 'mz', 'inv_ion_mobility', 'retention_time')


# Get a dict with data from frames 1, 5, and 67.
pprint(D.query(frames=[1,5,67], columns=all_columns))
#         frame  scan     tof  intensity           mz  inv_ion_mobility  retention_time
# 0           1    33  312260          9  1174.655791          1.601142        0.326492
# 1           1    34  220720          9   733.480941          1.600000        0.326492
# 2           1    34  261438          9   916.952389          1.600000        0.326492
# 3           1    36   33072          9   152.355651          1.597716        0.326492
# 4           1    36  242110          9   827.311421          1.597716        0.326492
# ...       ...   ...     ...        ...          ...               ...             ...
# 224732     67   917  192745         51   619.285021          0.600774        7.405654
# 224733     67   917  201838         54   655.343937          0.600774        7.405654
# 224734     67   917  205954         19   672.001670          0.600774        7.405654
# 224735     67   917  236501         57   802.160552          0.600774        7.405654
# 224736     67   917  289480         95  1055.203750          0.600774        7.405654
# 
# [224737 rows x 7 columns]




# Get a dict with each 10th frame, starting from frame 2, finishing on frame 1000.   
pprint(D.query(frames=slice(2,1000,10), columns=all_columns))
#         frame  scan     tof  intensity           mz  inv_ion_mobility  retention_time
# 0           2    33   97298          9   302.347671          1.601142        0.434706
# 1           2    33  310524          9  1165.327281          1.601142        0.434706
# 2           2    34  127985          9   391.984100          1.600000        0.434706
# 3           2    35  280460          9  1009.675130          1.598858        0.434706
# 4           2    37  329377         72  1268.626208          1.596575        0.434706
# ...       ...   ...     ...        ...          ...               ...             ...
# 669552    992   909  198994          9   643.956206          0.609747      106.710279
# 669553    992   909  282616          9  1020.466272          0.609747      106.710279
# 669554    992   912  143270          9   440.966974          0.606382      106.710279
# 669555    992   915  309328          9  1158.922133          0.603017      106.710279
# 669556    992   916  224410          9   749.264705          0.601896      106.710279
# 
# [669557 rows x 7 columns]


# Get all MS1 frames 
# pprint(D.query(frames=D.ms1_frames, columns=all_columns))
# ATTENTION: that's quite a lot of data!!! You might exceed your RAM.


# If you want to extract not every possible columnt, but a subset, use the columns argument:
pprint(D.query(frames=slice(2,1000,10), columns=('tof','intensity',)))
#            tof  intensity
# 0        97298          9
# 1       310524          9
# 2       127985          9
# 3       280460          9
# 4       329377         72
# ...        ...        ...
# 669552  198994          9
# 669553  282616          9
# 669554  143270          9
# 669555  309328          9
# 669556  224410          9
# 
# [669557 rows x 2 columns]
# 
# This will reduce your memory usage.


# Still too much memory used up? You can also iterate over frames:
it = D.query_iter(slice(10,100,10), columns=all_columns)
pprint(next(it))
#        frame  scan     tof  intensity           mz  inv_ion_mobility  retention_time
# 0         10    34  171284          9   538.225728          1.600000        1.293682
# 1         10    36   31282          9   148.904423          1.597716        1.293682
# 2         10    38  135057          9   414.288925          1.595433        1.293682
# 3         10    39  135446          9   415.533724          1.594291        1.293682
# 4         10    41  188048          9   601.058384          1.592008        1.293682
# ...      ...   ...     ...        ...          ...               ...             ...
# 11470     10   908   86027        215   272.343029          0.610869        1.293682
# 11471     10   908  304306          9  1132.219605          0.610869        1.293682
# 11472     10   913  207422          9   677.993343          0.605260        1.293682
# 11473     10   916   92814         13   290.222999          0.601896        1.293682
# 11474     10   916   95769         86   298.185400          0.601896        1.293682
# 
# [11475 rows x 7 columns]


pprint(next(it))
#       frame  scan     tof  intensity           mz  inv_ion_mobility  retention_time
# 0        20    33  359979         31  1445.637778          1.601142        2.366103
# 1        20    33  371758         10  1516.851302          1.601142        2.366103
# 2        20    34  170678          9   536.019344          1.600000        2.366103
# 3        20    37  187676          9   599.626478          1.596575        2.366103
# 4        20    38   12946          9   115.828412          1.595433        2.366103
# ...     ...   ...     ...        ...          ...               ...             ...
# 6555     20   915   18954          9   126.209155          0.603017        2.366103
# 6556     20   915  136901        111   420.206274          0.603017        2.366103
# 6557     20   915  137327         26   421.579263          0.603017        2.366103
# 6558     20   915  137500          9   422.137478          0.603017        2.366103
# 6559     20   916   96488          9   300.139081          0.601896        2.366103
# 
# [6560 rows x 7 columns]



# All MS1 frames, but one at a time
iterator_over_MS1 = D.query_iter(D.ms1_frames, columns=all_columns)
pprint(next(it))
#        frame  scan     tof  intensity          mz  inv_ion_mobility  retention_time
# 0         30    33  210334          9  689.957428          1.601142        3.440561
# 1         30    34   24628          9  136.421775          1.600000        3.440561
# 2         30    34   86106          9  272.547880          1.600000        3.440561
# 3         30    34  165263          9  516.505046          1.600000        3.440561
# 4         30    35  224302          9  748.800355          1.598858        3.440561
# ...      ...   ...     ...        ...         ...               ...             ...
# 13076     30   903  266788          9  942.579660          0.616478        3.440561
# 13077     30   905  164502          9  513.791592          0.614234        3.440561
# 13078     30   914   77215        125  249.976426          0.604139        3.440561
# 13079     30   914   78321         22  252.731087          0.604139        3.440561
# 13080     30   917   92056         20  288.197894          0.600774        3.440561
# 
# [13081 rows x 7 columns]

pprint(next(it))
#       frame  scan     tof  intensity           mz  inv_ion_mobility  retention_time
# 0        40    33  310182          9  1163.493906          1.601142        4.511882
# 1        40    33  362121         11  1458.460526          1.601142        4.511882
# 2        40    34  364148         76  1470.646985          1.600000        4.511882
# 3        40    35  364858         40  1474.927542          1.598858        4.511882
# 4        40    37    4059          9   101.290007          1.596575        4.511882
# ...     ...   ...     ...        ...          ...               ...             ...
# 7354     40   915  126467        100   387.276839          0.603017        4.511882
# 7355     40   915  130612        106   400.197513          0.603017        4.511882
# 7356     40   916  131559         77   403.179227          0.601896        4.511882
# 7357     40   916  270542          9   960.772728          0.601896        4.511882
# 7358     40   917  127329         88   389.946380          0.600774        4.511882
# 
# [7359 rows x 7 columns]



# The frame lasts a convenient time unit that well suits chromatography peak elution.
# What if you were interested instead in finding out which frames eluted in a given time 
# time of the experiment?
# For this reasone, we have prepared a retention time based query:
# suppose you are interested in all frames corresponding to all that eluted between 10 and 12
# minute of the experiment.
# Well, here goes nothing:
D.rt_query(10,12)
#         frame  scan     tof  ...           mz  inv_ion_mobility  retention_time
# 0          92    33  361758  ...  1456.283499          1.601142       10.086899
# 1          92    36   65738  ...   222.282248          1.597716       10.086899
# 2          92    41  308330  ...  1153.590878          1.592008       10.086899
# 3          92    43  123618  ...   378.518961          1.589725       10.086899
# 4          92    48   65346  ...   221.365052          1.584017       10.086899
# ...       ...   ...     ...  ...          ...               ...             ...
# 128129    109   914  138760  ...   426.214230          0.604139       11.910014
# 128130    109   914  142129  ...   437.210911          0.604139       11.910014
# 128131    109   914  144566  ...   445.252770          0.604139       11.910014
# 128132    109   916  138933  ...   426.775504          0.601896       11.910014
# 128133    109   917  373182  ...  1525.576529          0.600774       11.910014

# [128134 rows x 7 columns]



# Get numpy array with raw data in a given range 1:10
pprint(D[1:10])
# array([[     1,     33, 312260,      9],s
#        [     1,     34, 220720,      9],
#        [     1,     34, 261438,      9],
#        ...,
#        [     9,    913, 204042,     10],
#        [     9,    914, 358144,      9],
#        [     9,    915, 354086,      9]], dtype=uint32)


# Simple access to 'analysis.tdf'? Sure:
D.tables_names()
#                         name
# 0             GlobalMetadata
# 1        PropertyDefinitions
# 2             PropertyGroups
# 3            GroupProperties
# 4            FrameProperties
# 5              FrameMsMsInfo
# 6            CalibrationInfo
# 7                     Frames
# 8                   Segments
# 9                   ErrorLog
# 10             MzCalibration
# 11           TimsCalibration
# 12  DiaFrameMsMsWindowGroups
# 13       DiaFrameMsMsWindows
# 14          DiaFrameMsMsInfo

# Just choose a table now:
D.table2df('TimsCalibration')
#    Id  ModelType  C0   C1          C2         C3    C4  C5        C6          C7         C8           C9
# 0   1          2   1  917  213.599846  75.817289  33.0   1 -0.009066  135.436364  13.326079  1663.341457
# and there will be pandas all the way down. 


# Also, one table (Frames) is so important, that you can get it immediately: 
D.frames
#           Id         Time Polarity  ScanMode  ...  TimsCalibration  PropertyGroup  AccumulationTime  RampTime
# 0          1     0.326492        +         9  ...                1              1            99.953    99.953
# 1          2     0.434706        +         9  ...                1              1            99.953    99.953
# 2          3     0.540987        +         9  ...                1              1            99.953    99.953
# 3          4     0.648887        +         9  ...                1              1            99.953    99.953
# 4          5     0.756660        +         9  ...                1              1            99.953    99.953
# ...      ...          ...      ...       ...  ...              ...            ...               ...       ...
# 11548  11549  1243.494142        +         9  ...                1              1            99.953    99.953
# 11549  11550  1243.599171        +         9  ...                1              1            99.953    99.953
# 11550  11551  1243.707291        +         9  ...                1              1            99.953    99.953
# 11551  11552  1243.811222        +         9  ...                1              1            99.953    99.953
# 11552  11553  1243.919337        +         9  ...                1              1            99.953    99.953

# [11553 rows x 17 columns]

# And here are some plotting functions:
D.plot_TIC()
D.plot_peak_counts()

intensities_matrix, mz_bin_borders, inv_ion_mobility_bin_borders = \
    D.intensity_given_mz_inv_ion_mobility()

D.plot_intensity_given_mz_inv_ion_mobility(
    intensities_matrix, 
    mz_bin_borders,
    inv_ion_mobility_bin_borders
)