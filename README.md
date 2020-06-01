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
As you can see, the whole data-set contains 404 183 875 measured ions (peaks).
Each ion is described by the frame, tims' scan, time of flight index, and intensity.
There are 11 553 frames, and 918 possible scans.
Frames, scans, and time of flight indices can be translated into their physical equivalents.
Frames correspond to retention times, scans to ion mobilities, and time of flights to mass over charge ratios.
To visualize these dependencies, call (if you have matplotlib installed),
```{python}
D.plot_models()
``` 
which will result in something like this:
![](https://github.com/MatteoLacki/timspy/blob/devel/models.png "Comparing Human-Yeast-Ecoli Proteomes")

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

To install extra modules required for plotting (matplotlib, plotnine), use
```{bash}
pip install timspy[plots]
```
or install them manually with
```{bash}
pip install matplotlib plotnine timspy
```
which seems much less of a hastle than figuring out how pypi internals work.

# Too bloat?

Try double bloat! Joking, but we have a simpler module too.
Check out our [timsdata module](https://github.com/MatteoLacki/timsdata).
It is a very small wrapper around Bruker's SDK and makes it easier to extract data only in numpy.arrays or pure pythonic objects.