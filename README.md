# TimsPy

A data science friendly data access to timsTOF Pro mass spectrometry data.

### Requirements

TimsPy is compatible with Windows and Linux.
MacOS ain't supported.

### What gives?

Simple way to get data out of results collected with your Bruker timsTOF Pro from Python.
This definitely ain't no rocket science, but is pretty useful!

For example:

```{python}
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

As suggested by this quick data-view, the basic idea behind TimsPy is to represent the data in the convenient format of pandas data.frames and therefore facilitate simple data-exploration straight from Python.
As you can see, the whole data-set contains 404 183 875 measured ions (peaks).
Each ion is described by the frame, tims' scan, time of flight index, and intensity.
There are 11 553 frames, and 918 possible scans.
Frames, scans, and time of flight indices can be translated into their physical equivalents.
Frames correspond to retention times, scans to ion mobilities, and time of flights to mass over charge ratios.
To visualize these dependencies, call (if you have matplotlib installed),
```{python}
D.plot_models()
``` 
which results in
![](https://github.com/MatteoLacki/timspy/blob/devel/models.png "Comparing Human-Yeast-Ecoli Proteomes")

Now, TimsPy offers the possibility to easily obtain sub-sets of the very big data frame shown above.
The general convention for extracting data is `D[frames, scans]`.
For example, after calling `print(D)` you already know, that you can choose a scan from a range between 1 and 11553, and scans between 0 and 918.
Now, exploring first 100 frames and scans between 100 and 500 could be done as simple as by calling:
```{python}
X = D[1:101, 100:501]
print(X)
#      frame  scan     tof   i                                                                  
# 0        1   101  340222   9                                                                  
# 1        1   103   76314   9                                                                  
# ..     ...   ...     ...  ..                                                                  
# 396     99   499  233956  59                                                                  
# 397     99   499  300328   9                                                                  
#                                                                                               
# [283014 rows x 4 columns]
```
note the use of pythonic indexing of the right borders of selection ranges.
You do not have to explicitly precise the scans: if you don't, we will use all in range:
```{python}
X = D[1:101]
print(X)
#       frame  scan     tof   i                                                                 
# 0         1    33  312260   9
# 1         1    34  220720   9
# ...     ...   ...     ...  ..
# 8394     99   914  313598   9
# 8395     99   917  354548   9
# 
# [1626073 rows x 4 columns]
```
Also, you do not have to provide slices, any iterable will also do:
```{python}
X = D[[1000, 2300, 5000]]
print(X)
#        frame  scan     tof   i                                                                
# 0       1000    35  257054   9
# 1       1000    35  373510   9
# ...      ...   ...     ...  ..
# 40100   5000   914   71976  99
# 40101   5000   915  107966   9
# 
# [61365 rows x 4 columns]
```

Also, note that if you select to much data, you will run out of RAM.
That's not nice.
To by-pass this, for the time being, TimsPy offers the possibility to use Python iterators and explore the data one frame at a time.
```{python}
it = D.iter[:10000]
print(next(it))
#       frame  scan     tof    i
# 0         1    33  312260    9
# 1         1    34  220720    9
# ...     ...   ...     ...  ...
# 1599      1   915  233374    9
# 1600      1   916  335348   77
# 
# [1601 rows x 4 columns]

print(next(it))
#       frame  scan     tof   i
# 0         2    33   97298   9
# 1         2    33  310524   9
# ...     ...   ...     ...  ..
# 6596      2   913   56442   9
# 6597      2   915  172202   9
# 
# [6598 rows x 4 columns]
```
The `D.iter[...]` will accept anything `D[...]` would.
The idea is, that you can easily do your operations in loops and be happy.

### Physical units
You might not be entirely happy being stuck with the default non-physical indices that the SDK operates under the hood.
We find them very nice, as they keep data smaller, but we get it, you want to see the physics!

Get physical!
The general convention for extracting data in physical units is to extract data in rectangles `D.phys[min_rt:max_rt, min_im:max_im]`.
For example,
```{python}
X = D.phys[1:10. 0.7:1.5]
print(X)
#             rt        im           mz   i
# 0     1.078353  1.600000  1352.120899  72
# 1     1.078353  1.598858  1353.444087  34
# ...        ...       ...          ...  ..
# 7828  9.979100  0.611991   645.068941   9
# 7829  9.979100  0.610869  1365.561858   9
# 
# [1479453 rows x 4 columns]
```
and you can get iterators too, as simple as:
```{python}
it = D.physIter[1:10]
print(next(it))
#              rt        im           mz    i                                                   
# 0      1.078353  1.600000  1352.120899   72
# 1      1.078353  1.598858  1353.444087   34
# ...         ...       ...          ...  ...
# 12843  1.078353  0.604139   247.034346   90
# 12844  1.078353  0.603017    97.343304    9
# 
# [12845 rows x 4 columns]

print(next(it))
#              rt        im           mz   i                                                    
# 0      1.185108  1.600000  1362.910106  19
# 1      1.185108  1.600000  1371.681724  56
# ...         ...       ...          ...  ..
# 12261  1.185108  0.604139  1434.697864   9
# 12262  1.185108  0.603017  1410.652404   9
# 
# [12263 rows x 4 columns]
```

### DIA experiments

Data Independent Acquisition is a mode for performing the measurement in a consistent way, that assures a better, non-random coverage of the proteome.
In particular, on timsTOF Pro the MS2 (fragmented) data is collected on windows.
Want to see them? Don't use `TimsPyDF`, but a specialized subclass:
```{python}
p = '/path/to/brukers/folder/with/DIA/data.d')
D = TimsDIA(p)
D.plot_windows()
```
![](https://github.com/MatteoLacki/timspy/blob/devel/all_windows.png "DIA windows")
Too many windows?

```{python}
D.plot_windows("window_gr in [1,5,10]")
```
![](https://github.com/MatteoLacki/timspy/blob/devel/windows_1_5_10.png "DIA windows 1, 5, and 10")

Also, you might want to explore data in particular windows.
This is also supported.
```{python}
it = D.iter['window_gr == 1']
print(next(it))
#       frame  scan     tof   i
# 0         2    33   97298   9
# 1         2    33  310524   9
# ...     ...   ...     ...  ..
# 6596      2   913   56442   9
# 6597      2   915  172202   9
# 
# [6598 rows x 4 columns]
```
In fact, you can write queries involving any variables in tables `D.windows` and `D.frames`.

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
We are working on support for the vaex module on several levels.
Even when you sleep, our team of commited coder(s) is working his (their) way through the unknown.

* Specialized methods for DDA experiments.
* Better C++ integration
* Intergration with vaex as a lazy alternative to pandas


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

To install extra modules required for plotting (matplotlib, plotnine), use
```{bash}
pip install timspy[plots]
```
or install them manually with
```{bash}
pip install matplotlib plotnine timspy
```
which seems much less of a hastle than figuring out how pypi internals work.

### Too bloat?

We have a simpler module too.
Check out our [timsdata module](https://github.com/MatteoLacki/timsdata).
It is a very small wrapper around Bruker's SDK and makes it easier to extract data only in numpy.arrays or pure pythonic objects.