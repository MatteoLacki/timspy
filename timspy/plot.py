"""Basic plotting procedures."""


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


def plot2d(X, x_axis, x_i, x_l, y_axis, y_i, y_l, show=True, **kwds):
    import matplotlib.pyplot as plt
    plt.imshow(X, origin='lower', cmap='inferno', aspect='auto', **kwds)
    plt.xticks(x_i, x_l)
    plt.yticks(y_i, y_l)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()
    if show:
        plt.show()