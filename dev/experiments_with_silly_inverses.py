yinv = y[::-1]
im = [1.601, 1.60001]
im = y[K]
K = len(yinv) - np.searchsorted(yinv, im, side='right')
y[K]   # leq

K = len(yinv) - np.searchsorted(yinv, im, side='left')
y[K-1] # geq


D.scan2im((D.min_scan, D.max_scan))
D.min_im
D.max_im

def which_min_geq(x, y):
    return np.searchsorted(x, y, side='left')

def which_max_leq(x,y):
    return np.searchsorted(x, y, side='right')-1

# border cases??
x = np.array([1, 2.1, 2.2, 3])
x[which_min_geq(x, 3)]
x[which_max_leq(x, 154.0)]

D.im2scan([1.14])


K = len(yinv) - np.searchsorted(yinv, im, side='right')
np.which_min(0,)