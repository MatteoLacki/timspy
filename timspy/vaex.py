"""

Log:
TimsVaex cannot be a subclass vaex.dataset.HDF5MappableStorage because of potential clashes between column names and methods.
"""

import functools
import numpy as np
import pathlib

from .sql import tables_names, table2df
from .plot import plot2d


round10 = lambda x: 10**np.floor(np.log10(x)).astype(int)


class TimsVaex(object):
    def __init__(self, path2hdf, path2tdf):
        """
        Args:
            path2hdf (str): Path to the hdf5 stored timsTOF dataset.
            path2tdf (str): Path to the 'analysis.tdf' sqlite3 DB.
        """
        import vaex

        self.path2hdf = pathlib.Path(path2hdf)
        self.path2tdf = pathlib.Path(path2tdf)
        self.df = vaex.open(str(self.path2hdf))
        self.columns = tuple(self.df.columns)
        self.frames = self.table2df('frames')
        self.min_frame = self.frames.Id.min()
        self.max_frame = self.frames.Id.max()
        self.frames_no = self.max_frame-self.min_frame+1
        self._ms1_mask = self.frames.MsMsType.values == 0
        self.ms1_frames = self.frames.Id[self._ms1_mask].values
        self.retention_time = self.frames.Time


    def tables_names(self):
        """List names of tables in the SQLite db.

        Returns:
            pd.DataTable: table with names of tables one can get with 'table2df'.
        """
        return tables_names(self.path2tdf)


    def table2df(self, name):
        """Retrieve a table with SQLite connection from a data base.

        Args:
            name (str): Name of the table to extract.
        Returns:
            pd.DataFrame: required data frame.
        """
        return table2df(self.path2tdf, name)

    def __repr__(self):
        return f"{self.__class__.__name__}.df\n{repr(self.df)}"

    @functools.lru_cache(maxsize=10)
    def minmax(self, column):
        if column == 'frame':
            return self.min_frame, self.max_frame 
        else:
            return self.df[column].minmax()
 
    def min(self, column):
        return self.minmax(column)[0]

    def max(self, column):
        return self.minmax(column)[1]

    @functools.lru_cache(2)
    def intensity_per_frame(self, recalibrated=True):
        if recalibrated:
            return self.frames.SummedIntensities.values 
        else:
            import vaex
            frame2summedIntensity = self.df.groupby(by='frame',
             agg={'SummedIntensity': vaex.agg.sum('intensity')}).to_pandas_df().sort_values('frame')
            return frame2summedIntensity.SummedIntensity.values

    def plot_TIC(self, recalibrated=True, show=True):
        """Plot peak counts per frame.

        Arguments:
            recalibrated (bool): Use Bruker recalibrated total intensities or calculate them from scratch with OpenTIMS?
            show (bool): Show the plot immediately, or just add it to the canvas?
        """
        import matplotlib.pyplot as plt
        MS1 = self._ms1_mask
        I = self.intensity_per_frame(recalibrated)
        plt.plot(self.retention_time[ MS1], I[ MS1], label="MS1")
        plt.plot(self.retention_time[~MS1], I[~MS1], label="MS2")
        plt.legend()
        plt.xlabel("Retention Time")
        plt.ylabel("Intensity")
        plt.title("Total Intensity [Ion Current]")
        if show:
            plt.show()

    def intensity_given_mz_inv_ion_mobility(self,
                                            frames=None,
                                            mz_bin_borders=np.linspace(500, 2500, 1001),
                                            inv_ion_mobility_bin_borders=np.linspace(0.8, 1.7, 101)):
        if frames is None:
            frames = self.ms1_frames




    # @property
    # @functools.lru_cache(1)
    # def TIC_frame_scan(self):
    #     f,F = self.frame_minmax
    #     s,S = self.scan_minmax
    #     return self.df.sum(self.df.intensity,
                           # binby=[self.df.scan, self.df.frame],
                           # limits=[[s-.5,S+.5], [f-.5, F+.5]],
                           # shape=[S-s+1, F-f+1])

    # def TIC_plot(self, show=True):
    #     import matplotlib.pyplot as plt
    #     tic = self.TIC()
    #     plt.plot(self.ms1_frames, tic[self.F.MsMsType == 0], label='MS1')
    #     plt.plot(self.ms2_frames, tic[self.F.MsMsType != 0], label='MS2')
    #     plt.legend()
    #     plt.tight_layout()
    #     if show:
    #         plt.show()

    # def TIC_frame_scan_plot_MS1(self, show=True):
    #     s, S = self.scan_minmax
    #     I = self.TIC_frame_scan[:, self.ms1_frames-1]
    #     x_i = np.arange(0, I.shape[1], round10(I.shape[1]), dtype=int)
    #     x_l = self.ms1_frames[x_i]
    #     y_i = np.arange(0, I.shape[0], round10(I.shape[0]), dtype=int)
    #     y_l = np.arange(s, S, round10(I.shape[0]), dtype=int)
    #     plot2d(I, 'Frame', x_i, x_l, 'Scan', y_i, y_l, show)

    # def TIC_frame_scan_plot_MS2(self, show=True):
    #     s, S = self.scan_minmax
    #     I = self.TIC_frame_scan[:, self.ms2_frames-1]
    #     x_i = np.arange(0, I.shape[1], round10(I.shape[1]), dtype=int)
    #     x_l = self.ms2_frames[x_i]
    #     y_i = np.arange(0, I.shape[0], round10(I.shape[0]), dtype=int)
    #     y_l = np.arange(s, S, round10(I.shape[0]), dtype=int)
    #     plot2d(I, 'Frame', x_i, x_l, 'Scan', y_i, y_l, show)
