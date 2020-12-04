# TimsPy

A data science friendly data access to timsTOF Pro mass spectrometry data.

### Requirements

TimsPy works well on Linux.
On Windows, it works with Python <= 3.7.3 due to changes in the distribution policy of the dlls by cpython.
This is currently being patched.
MacOS ain't supported.

### What can you expect?

Simple way to get data out of results collected with your Bruker timsTOF Pro from Python.
This definitely ain't no rocket science, but is pretty useful!

For example:

```python
from timspy.timspy import TimspyDF

p = '/path/to/brukers/folder/with/data.d')
D = TimsPyDF(p)
print(D)
                                                                                    
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



### Basic plotting
You can plot the overview of your experiment.
To do that, select the minimal and maximal frames to plot,
```{python}
D.plot_overview(1000, 4000)
```
![](https://github.com/MatteoLacki/timspy/blob/devel/overview.png "DIA experiment overview")

Also, you can plot peak counts
```{python}
D.plot_peak_counts()
```
![](https://github.com/MatteoLacki/timspy/blob/devel/peak_counts.png "Counts of peaks in MS1 and MS2 experiments.")

In orange, you see the sum of intensities for each frame (Total Ion Current) for MS1 frames, and in gray for MS2 frames.

These are not fast methods.
From our experience, it's much faster to save data to hdf5 format and then use vaex to plot it interactively.
This will be coming up soon, so stay tuned!

### Plans
* specialized methods for DDA experiments
* going fully open-source

### Installation

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

To install vaex support, use
```{bash}
pip install timspy[vaex]
```
or install add in additional modules
```{bash}
pip install vaex-core vaex-hdf5
```
which seems much less of a hastle than figuring out how pypi internals work.

### API documentation

Please [visit our documentation page](https://matteolacki.github.io/timspy/index.html).

### Too bloat?

We have a simpler module too.
Check out our [timsdata module](https://github.com/michalsta/opentims).

Best wishes,

Matteo Lacki & Michal (Startrek) Startek