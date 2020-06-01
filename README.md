### TimsPy

A data science friendly data access to timsTOF Pro mass spectrometry data.

# Requirements

TimsPy is compatible with Windows and Linux.
MacOS ain't supported.

# What gives?

Simple way to get data out of results collected with your Bruker timsTOF Pro from Python.
For example:

```{python}
from timspy.timspy import TimspyDF

p = '/path/to/brukers/folder/with/data.d')
D = TimsPyDF(p)
print(D)

# Out[9]:                                                                                       
#            frame  scan     tof  i                                                             
# 0              1    33  312260  9                                                             
# 1              1    34  220720  9                                                             
# 2              1    34  261438  9                                                             
# .................................                                                             
# 404183873  11553   911  295164  9                                                             
# 404183874  11553   915   82016  9                                                             
# 404183875  11553   916  303778  9                                                             
# 
# [frames 1:11553, scans 0:918]           
```

As suggested by this quick data-view, the basic idea behind TimsPy is to represent the data in the convenient format of pandas data.frames.


# Installation

```{bash}
pip install timspy
```
or for devel version:
```{bash}
pip install -e git+https://github.com/MatteoLacki/timspy/tree/devel
```
or with git:
```{bash}
git clone https://github.com/MatteoLacki/timspy
cd timspy
pip install -e .
```

# Too bloat?

Try double bloat! Joking, but we have a simpler module too.
Check out our [timsdata module](https://github.com/MatteoLacki/timsdata).
It is a very small wrapper around Bruker's SDK and makes it easier to extract data only in numpy.arrays or pure pythonic objects.