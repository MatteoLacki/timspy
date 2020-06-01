"""Model is like an order of magnitude faster than python dictionary."""
%load_ext autoreload
%autoreload 2
from pathlib import Path
import numpy as np
from numpy import random

from timspy.timspy import TimspyDF

p = Path('/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')
D = TimspyDF(p) # timsdata does not support :

XX = D[1000]
M = D.scan2im_model
x = M.x
y = M.y

yinv = y[::-1]
X = random.rand(len(XX))
X *= (y[0]-y[-1])
X += y[-1]

%%timeit
np.searchsorted(yinv, X) # FASTER THAN DICT, SLOWER THAN THE POLYNOMIAL MODEL
len(X)

scan2im = dict(zip(D.scan2im_model.x, D.scan2im_model.y))

%%timeit
XX.scan.map(scan2im)

%%timeit
D.scan2im_model(XX.scan) #FASTEST

# todo: change the bloody search sorted into a real algorithm.
# compare it with simple list iteration: because the results can be simpy done in groups
# enough for today.



