"""
We made a class to simplify access to DIA data.
Its parent class is TimsPyDF, so you can use all of its methods.
"""
from timspy.dia import TimsPyDIA



path = 'path/to/your/data.d'
D = TimsPyDF(path) # get data handle

# Interested in Windows?
D.windows
#      WindowGroup  ScanNumBegin  ScanNumEnd  ...  CollisionEnergy      mz_left     mz_right
# 0              1             0          36  ...             10.0  1265.384615  1301.384615
# 1              1            37          73  ...             10.0  1220.512821  1256.512821
# 2              1            74         110  ...             10.0  1175.641026  1211.641026
# 3              1           111         146  ...             10.0  1130.769231  1166.769231
# 4              1           147         183  ...             10.0  1085.897436  1121.897436
# ..           ...           ...         ...  ...              ...          ...          ...
# 520           21           735         771  ...             10.0   607.948718   643.948718
# 521           21           772         807  ...             10.0   563.076923   599.076923
# 522           21           808         844  ...             10.0   518.205128   554.205128
# 523           21           845         881  ...             10.0   473.333333   509.333333
# 524           21           882         918  ...             10.0   428.461538   464.461538

# [525 rows x 8 columns]



# Interested which frame collects data from which window group?
D.frames_meta()
#           Id         Time Polarity  ScanMode  ...  IsolationWidth  CollisionEnergy      mz_left     mz_right
# 0          1     0.326492        +         9  ...             NaN              NaN          NaN          NaN
# 1          2     0.434706        +         9  ...            36.0             10.0  1265.384615  1301.384615
# 2          3     0.540987        +         9  ...            36.0             10.0  1220.512821  1256.512821
# 3          4     0.648887        +         9  ...            36.0             10.0  1175.641026  1211.641026
# 4          5     0.756660        +         9  ...            36.0             10.0  1130.769231  1166.769231
# ...      ...          ...      ...       ...  ...             ...              ...          ...          ...
# 11548  11549  1243.494142        +         9  ...            36.0             10.0  1460.512821  1496.512821
# 11549  11550  1243.599171        +         9  ...            36.0             10.0  1415.641026  1451.641026
# 11550  11551  1243.707291        +         9  ...            36.0             10.0  1370.769231  1406.769231
# 11551  11552  1243.811222        +         9  ...            36.0             10.0  1325.897436  1361.897436
# 11552  11553  1243.919337        +         9  ...            36.0             10.0  1281.025641  1317.025641

# [11553 rows x 26 columns]
# This function joins the proper tables for you, so you don't have to.
# That how good that function is.