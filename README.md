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
The general convention for extracting data is `D[frames, scans]`:
For example, after calling `print(D)` you already know, that you can choose a scan from a range between 1 and 11553, and scans between 0 and 918.
Now, exploring first 100 frames and scans between 100 and 500 could be done as simple as by calling:
```{python}
X = D[1:101, 100:501]
print(X)
#      frame  scan     tof   i                                                                  
# 0        1   101  340222   9                                                                  
# 1        1   103   76314   9                                                                  
# 2        1   103  267974   9                                                                  
# 3        1   104  285432   9                                                                  
# 4        1   105   46454   9                                                                  
# ..     ...   ...     ...  ..                                                                  
# 393     99   499  230044  52                                                                  
# 394     99   499  230710  22                                                                  
# 395     99   499  233908  45                                                                  
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
# 2         1    34  261438   9
# 3         1    36   33072   9
# 4         1    36  242110   9
# ...     ...   ...     ...  ..
# 8391     99   913  200278   9
# 8392     99   914  101230  65
# 8393     99   914  257396   9
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
# 2       1000    44  251548   9
# 3       1000    46  104446   9
# 4       1000    51  110434   9
# ...      ...   ...     ...  ..
# 40097   5000   909  319446   9
# 40098   5000   911  173796   9
# 40099   5000   912  155300   9
# 40100   5000   914   71976  99
# 40101   5000   915  107966   9
# 
# [61365 rows x 4 columns]
```


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