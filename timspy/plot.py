"""Basic plotting procedures."""
import pandas as pd

def plot_spectrum(mz, intensity, show=True, **kwds):
    """A simple stem-plot stick spectrum.
    
    You need to have matplotlib installed for this method to work.
    
    Arguments:
        MZ (iterable): mass over charge values.
        I (iterable): intensities corresponding to mass over charge ratios.
        show (bool): show the plot
    """
    import matplotlib.pyplot as plt
    plt.stem(mz, intensity, markerfmt=' ', use_line_collection=True, **kwds)
    if show:
        plt.show()


def plot3d(x, y, z, show=True, **kwds):
    """Make a 3D plot of data."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.scatter(ax, xs=x, ys=y, zs=z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if show:
        plt.show()


def heatmap(X,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            interpolation='none',
            aspect='auto',
            cmap='inferno',
            origin='lower',
            show=True,
            **kwds):
    """Plot a heatmap.

    imshow with a selection of more natural defaults for mass spectrometry.

    Arguments:
        X (np.array): 2D array with heatmap data.
        x_bin_borders (np.array): Positions of bin borders for rows of X.
        y_bin_borders (np.array): Positions of bin borders for columns of X.
        interpolation (str): Type of interpolation used in 'matplotlib.pyplot.imshow'.
        aspect (str): Aspect ratio in 'matplotlib.pyplot.imshow'.
        cmap (str): Color scheme for the 'matplotlib.pyplot.imshow'.
        origin (str): Where should the origin of the coordinate system start? Defaults to bottom-left. Check 'matplotlib.pyplot.imshow'. 
        show (bool): Show the plot immediately, or just add it to the canvas?
        **kwds: Keyword arguments for 'matplotlib.pyplot.imshow' function.
    """
    import matplotlib.pyplot as plt

    X_is_df = isinstance(X, pd.DataFrame)

    if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
        extent = [x_min, x_max, y_min, y_max]
    elif X_is_df:
        extent = [X.index[0], X.index[-1], X.columns[0], X.columns[-1]]
    else:
        extent = None

    plt.imshow(X.T,
               extent=extent,
               interpolation=interpolation,
               aspect=aspect,
               cmap=cmap,
               origin=origin,
               **kwds)
    if X_is_df:
        plt.xlabel(X.index.name)
        plt.ylabel(X.columns.name)
    if show:
        plt.show()


def abline(slope, intercept, **kwds):
    """Plot a line from slope and intercept"""
    import matplotlib.pyplot as plt
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', **kwds)


def abline_minmax_x(slope, intercept, min_x, max_x, **kwds):
    """Plot a line from slope and intercept"""
    import matplotlib.pyplot as plt
    axes = plt.gca()
    x_vals = [min_x, max_x]
    y_vals = [intercept + slope*x for x in x_vals]
    plt.plot(x_vals, y_vals, '--', **kwds)