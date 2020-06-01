import numpy as np


#TODO: replace this with a native non-numpy solution.
def weighted_quantile(x, q, w=None, x_sorted=False):
    """Calculate weighted quantiles.

    Code taken from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    stripped down to our needs.

    Args:
        x (iterable): values in the linear space.
        q (iterable): quantiles to calculate (between 0 and 1).
        w (iterable): weights related to each observation in x.
        x_sorted (boolean): are observations in x sorted?

    Returns:
        np.array: computed quantiles.
    """
    x = np.array(x)
    q = np.array(q)
    if w is None:
        w = np.ones(len(x))
    w = np.array(w)
    assert np.all(q >= 0) and np.all(q <= 1), 'q should be in [0, 1]'
    if not x_sorted:
        sorter = np.argsort(x)
        x = x[sorter]
        w = w[sorter]
    weighted_q = np.cumsum(w) - 0.5 * w
    weighted_q /= np.sum(w)
    return np.interp(q, weighted_q, x)


def agg_mz_inc_spec(mz, I, round_dig=0):
    """Aggregate rounded m/z values (mz assumed sorted)."""
    mz = np.round(mz)
    mz_ = mz[0]
    I_ = 0.0
    for _mz, _I in zip(mz, I):
        if mz_ == _mz:
            I_ += _I
        else:
            yield mz_, I_
            I_ = _I
            mz_ = _mz
    yield mz_, I_


def which_min_geq(x, y):
    return np.searchsorted(x, y, side='left')


def which_max_leq(x,y):
    return np.searchsorted(x, y, side='right')-1
