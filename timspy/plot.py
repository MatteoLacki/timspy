def plot_spectrum(MZ, I, show=True):
    """A simple stem-plot stick spectrum.
    
    You need to have matplotlib installed for this method to work.
    
    Args:
        MZ (iterable): mass over charge values.
        I (iterable): intensities corresponding to mass over charge ratios.
        show (bool): show the plot
    """
    import matplotlib.pyplot as plt
    plt.stem(MZ, I)
    if show:
        plt.show()


def plot3d(x, y, z, show=True, **kwds):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.scatter(ax, xs=x, ys=y, zs=z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if show:
        plt.show()
