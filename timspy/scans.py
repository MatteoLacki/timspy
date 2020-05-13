import pandas as pd


class Scans(pd.DataFrame):
    def __init__(self, S, frames, min_scan, max_scan):
        super().__init__(S)
        self.columns = range(min_scan, max_scan)
        self.index = frames

    def plot(self, show=True, **plt_kwds):
        """Show number of peaks found in each scan in each frame, or the number of non-empty scans in each frame.

        binary (boolean): plot only scan usage.
        """
        import matplotlib.pyplot as plt
        plt.imshow(self.T, **plt_kwds)
        if show:
            plt.show()

    def plot1d(self, binary=False, color='orange', show=True, **vlines_kwds):
        import matplotlib.pyplot as plt

        if binary:
            plt.axhline(y=min(self.columns), color='r', linestyle='-')
            plt.axhline(y=max(self.columns), color='r', linestyle='-')
        S = np.count_nonzero(self, axis=1) if binary else self.sum(axis=1)
        if 'color' not in vlines_kwds:
            vlines_kwds['color'] = color
        plt.vlines(S.index,0, S, **vlines_kwds)
        if show:
            plt.show()

