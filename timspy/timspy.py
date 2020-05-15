"""Wrappers for TimsData with more DataScientistFriendly functions."""

from functools import lru_cache
from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import vaex as vx

from timsdata import TimsData
from timsdata.slice_ops import parse_idx
from rmodel.polyfit import polyfit

from .iterators import ranges
from .sql import table2df


class AdvancedTims(TimsData):
    """TimsData that uses info about Frames."""
    def __init__(self, analysis_directory, use_recalibrated_state=False, **model_args):
        """Create an instance of the AdvancedTims class.

        Args:
            analysis_directory (str,Path): The '*.d' folder with the '*.tdf' files.
            use_recalibrated_state (bool): No idea yet.
        """
        super().__init__(analysis_directory, use_recalibrated_state=False)
        self.frames = self.table2df('Frames').rename(columns={'Time':'rt',
                                                    'Id':'frame'}).sort_values('frame').set_index('frame')
        self.min_frame = self.frames.index.min()
        self.max_frame = self.frames.index.max()
        self.border_frames = self.min_frame, self.max_frame 
        self.min_scan = 0
        max_scan = self.frames.NumScans.unique()
        if len(max_scan) != 1:
            raise RuntimeError("Number of TIMS pushes is not constant. This is not the class you want to instantiate on this data.")
        self.max_scan = max_scan[0]
        self.border_scans = self.min_scan, self.max_scan
        self.fit_frame2rt_model()
        self.fit_scan2im_model()
        self.fit_mzIdx2mz_model()


    def _get_mz_frames(self, min_mz=100, max_mz=4000, mass_indices_cnt=1000, frames_no=40):
        minIdx, maxIdx = self.mzToIndex(1, [min_mz, max_mz]).astype(int)
        massIndices = np.linspace(minIdx, maxIdx, mass_indices_cnt).astype(int)
        frames      = np.linspace(self.min_frame, self.max_frame, frames_no).astype(int)
        indexToMz   = np.vectorize(self.indexToMz, signature='(),(i)->(i)')
        return indexToMz(frames, massIndices)


    def _estimate_T1_impact(self, min_mz=100, max_mz=4000, mass_indices_cnt=1000, frames_no=40):
        """Estimate models mz[f,idx] = mz[1,idx] ( T1[f]/T1[1] )**B.

        The model is estimated independently for a selection of mass over charge ratios.
        Each mass over charge ratio correspond to a mass index.
        We set up a grid of 'mz_cnt' values in the mass index domain.
        The lowest mass index corresponds to 'min_mz', and the highest to 'max_mz'.
        For each mass index, we estimate the relationship on a grid of frame numbers.
        """
        logMZ = np.log(self._get_mz_frames(min_mz, max_mz, mass_indices_cnt, frames_no))
        T1 = self.frames.T1.loc[frames].values
        lognorm_T1 = np.log(T1) - np.log(T1[0])
        return lognorm_T1.dot( np.subtract(logMZ, logMZ[0,:]) )


    def _absolute_difference_of_mz(self, min_mz=100, max_mz=4000, mass_indices_cnt=1000, frames_no=40):
        """Find the maximal difference between m/z for diffent mass indices for different frames."""
        MZ = self._get_mz_frames(min_mz, max_mz, mass_indices_cnt, frames_no)
        return np.abs(MZ - MZ[0,:]).max()


    def _absolute_difference_of_im(self, frames_no=40):
        """Find the maximal difference between im for diffent scans for different frames."""
        frames = np.linspace(self.min_frame, self.max_frame, frames_no).astype(int)
        scans = np.linspace(self.min_scan, self.max_scan, frames_no).astype(int)
        scan2im = np.vectorize(self.scanNumToOneOverK0, signature='(),(i)->(i)')
        IM = scan2im(frames, scans)
        return np.abs(IM - IM[0,:]).max()        


    def frames_no(self):
        """Return the number of frames.

        Returns:
            int: Number of frames.
        """
        return len(self.frames)


    @lru_cache(maxsize=1)
    def MS1_frameNumbers(self):
        """Get the numbers of frames in MS1.
        
        Returns:
            np.array: numbers of frames in MS1.
        """
        F = self.frames
        return F.index.get_level_values('frame')[F.MsMsType == 0].unique().values


    @lru_cache(maxsize=1)
    def MS2_frameNumbers(self):
        """Get the numbers of frames in MS2.
        
        Returns:
            np.array: numbers of frames in MS2.
        """
        F = self.frames
        return F.index.get_level_values('frame')[F.MsMsType == 9].unique().values


    def frame2rt(self, frames):
        """Translate frame number to a proper retention time.

        Args:
            frames (int,list): frame numbers to translate.
        """
        frames = np.array(frames) + 1
        return self.frames.rt.values[frames]


    def rt2frame(self, rts):
        """Translate frame number to a proper retention time.

        Args:
            frames (int,list): frame numbers to translate.
        """
        rts = np.array(rts)
        assert np.logical_and(0 <= rts, rts <= self.frames.rt.max()).all(), "retention times out of range of the experiment."
        return np.searchsorted(self.frames.rt, rts)+1


    def __repr__(self):
        return f"{self.__class__.__name__}({self.frames_no()} frames)"


    @lru_cache(maxsize=1)
    def global_TIC(self):
        """Get the Total Ion Current across the entire experiment.

        Returns:
            int: Total Ion Current value.
        """
        s,S = self.border_scans
        f,F = self.border_frames
        return sum(self.frameTIC(f, s, S, True) for f in range(f, F+1))


    @lru_cache(maxsize=1)
    def count_all_peaks(self):
        """Count all the peaks in the database.

        Returns:
            int: number of peaks.
        """
        s, S = self.border_scans
        f, F = self.border_frames
        return int(sum(self.count_peaks(f,s,S) for f in range(f, F+1)))


    def fit_mzIdx2mz_model(self, mz_min=0, mz_max=3000, deg=2, prox_grid_points=1000):
        """Get a model that translates mass indices (time) into mass to charge ratios.

        Args:
            mz_min (float): minimal mass over charge.
            mz_max (float): maximal mass over charge.
            prox_grid_points (int): Approximate number of points for fitting.
    
        Return:
            Polyfit1D: a fitted 1D polynomial.
        """
        mzIdx_min, mzIdx_max = self.mzToIndex(1, [mz_min, mz_max]).astype(int)
        mzIdx_step = (mzIdx_max - mzIdx_min) // 1000
        mzIdx = np.arange(mzIdx_min, mzIdx_max, mzIdx_step)
        mz = self.indexToMz(1, mzIdx)
        self.mzIdx2mz_model = polyfit(mzIdx, mz, deg=deg)


    def fit_frame2rt_model(self, deg=5):
        """Fit a model that will change frame numbers to retention time values."""
        fr = np.arange(self.min_frame, self.max_frame+1)
        rt = self.frames.rt.values
        self.frame2rt_model = polyfit(fr, rt, deg=deg)


    def fit_scan2im_model(self, deg=4):
        scans = np.arange(self.min_scan, self.max_scan+1)
        ims = self.scan2im(scans)
        self.scan2im_model = polyfit(scans, ims, deg=deg)


    @lru_cache(maxsize=64)
    def table2df(self, name):
        """Retrieve a table with a given name from the '*.tdf' file.

        Args:
            name (str): The name of the table to retrieve.        
        """
        return table2df(self.conn, name)


    def mzIdx2mz(self, mz_idx, frame=1):
        """Translate mass indices (flight times) to mass over charge ratios.

        Args:
            mz_idx (int,iterable,np.array,pd.Series): mass indices.
            frame (integer): for which frame this calculations should be performed. These are very stable across
        """
        return self.indexToMz(frame, mz_idx)


    def mz2mzIdx(self, mz, frame=1):
        """Translate mass over charge ratios to mass indices (flight times).

        Args:
            mz (int,iterable,np.array,pd.Series): mass to charge ratios.
            frame (integer): for which frame this calculations should be performed. These are very stable across
        """
        return self.mzToIndex(frame, mz).astype(np.uint32)


    def scan2im(self, scan, frame=1):
        """Translate scan numbers to ion mobilities.

        Args:
            scan (int,iterable,np.array,pd.Series): scans.
            frame (integer): for which frame this calculations should be performed. These do not change accross the experiments actually.
        """
        return self.scanNumToOneOverK0(frame, scan)


    def im2scan(self, im, frame=1):
        """Translate ion mobilities to scan numbers.

        Args:
            im (int,iterable,np.array,pd.Series): ion mobilities.
            frame (integer): for which frame this calculations should be performed. These do not change accross the experiments actually.
        """
        return self.oneOverK0ToScanNum(frame, im).astype(np.uint32)


    def frame_scan_mzIdx_I_df(self, frame, scan_begin, scan_end):
        """Get a data frame with measurements for a given frame and scan region.

        The output data frame contains four columns: first repeats the frame number,
        second contains scan numbers, third contains mass indices, and the last contains intensities.
        
        Args:
            frame (int, iterable, slice): Frames to output.
            scan_begin (int): Lower scan.
            scan_end (int): Upper scan.
        Returns:
            pandas.DataFrame: four-columns data frame.
        """
        out = pd.DataFrame(self.frame_array(frame, scan_begin, scan_end))
        out.columns = ('frame', 'scan', 'mz_idx','i')
        return out


    def plot_models(self, show=True):
        """Plot model fittings.

        Args:
            show (boolean): Show the plot or only append it to the current canvas.
        """
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.sca(ax1)
        self.mzIdx2mz_model.plot(show=False, label='Mass Index vs M/Z')
        plt.legend()
        plt.sca(ax2)
        self.frame2rt_model.plot(show=False, label='Frame vs Retention Time')
        plt.legend()
        plt.sca(ax3)
        self.scan2im_model.plot(show=False, label='Scan No vs Drift Time')
        plt.legend()
        if show:
            plt.show()


    def iter_dfs(self, *args, **kwds):
        """Iterate over data frames with TDF data.
        
        Arguments like for 'self.get_frame_scanMin_scanMax'.

        Yields:
            A flow of numpy arrays.
        """
        for A in self.iter_arrays(*args, **kwds):
            F = pd.DataFrame(A)
            F.columns = ('frame', 'scan', 'mz_idx', 'i')
            yield F


    def __getitem__(self, x):
        if isinstance(x, tuple):
            if len(x) == 2:
                if isinstance(x[0], str):
                    frames = self.frames.query(x[0]).index.get_level_values('frame')
                    x = (frames, x[1])
                arr = super().__getitem__(x)
            else:
                raise Warning('Strange arguments.')
        else:
            s, S = self.border_scans
            arr = super().__getitem__((x, slice(s,S)))
        res = pd.DataFrame(arr)
        res.columns = 'frame', 'scan', 'mz_idx', 'i'
        return res


    def array(self, frames, filter_frames=''):
        """Return a numpy array with given data.

        Arguments like for 'self.get_frame_scanMin_scanMax'.

        Returns:
            np.array: an array with four columns: frame, scan, mass index (flight time), and intensity.
        """
        if filter_frames:
            frames = np.r_[self.frames.loc[frames].query(filter_frames).index.get_level_values('frame')]
        else:
            frames = np.r_[frames]
        if len(frames) == 0:
            return np.empty(shape=(0,4))
        else:
            return np.concatenate(list(self.iter_arrays(frames)), axis=0)


    def df(self, *args, **kwds):
        """Return a data frame with a selection of data.

        Arguments like for 'self.array'.

        Returns:
            pandas.DataFrame: with four columns: frame, scan, mass index (flight time), and intensity.
        """
        df = pd.DataFrame(self.array(*args,**kwds))
        df.columns = ('frame', 'scan', 'mz_idx', 'i')
        return df


    def plot_overview(self, min_frame, max_frame, show=True):
        """Plot overview."""
        import matplotlib.pyplot as plt

        df = self[min_frame:max_frame]
        df['mz'] = self.mzIdx2mz(df.mz_idx)
        df['im'] = self.scan2im(df.scan)
        X = df.groupby([round(df.im, 2), round(df.mz)]).i.sum().reset_index()
        im_res = len(X.im.unique())
        mz_res = len(X.mz.unique())
        plt.hist2d(X.mz, X.im, weights=X.i, bins=(mz_res, im_res), cmap=plt.cm.magma)
        del df
        if show:
            plt.show()


    def to_hdf5(self, out_folder, min_frame=None, max_frame=None, step=1000):
        """Dump project project to hdf5.

        Args:
            out_folder (str): Path to the folder where to store the outcome.
            min_frame (int): Minimal frame in selection for saving.
            max_frame (int): Maximal frame in selection for saving.
        """
        min_frame = self.min_frame if min_frame is None else min_frame
        max_frame = self.max_frame if max_frame is None else max_frame
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        types_remap = {'frame':'uint16', 'scan':'uint16', 'mz_idx':'uint32', 'i':'uint32'}
        #TODO: types should be remaped earlier!
        for f, F in ranges(min_frame, max_frame+1, step):
            pd_df = self[f:F]
            pd_df = pd_df.astype(types_remap)
            path = str(out_folder/f"{f}_{F}.hdf5")
            vx_df = vx.from_pandas(pd_df, copy_index=False)
            vx_df.export_hdf5(path=path)
            del vx_df, pd_df


    @lru_cache(maxsize=1)
    def count_all_peaks(self):
        """Get the number of peaks detected per each (frame,scan)."""
        frames = range(self.min_frame, self.max_frame+1)
        s, S = self.border_scans
        return pd.DataFrame(np.vstack([self.peakCnts_massIdxs_intensities(f,s,S)[s:S]
                                       for f in frames]), index=frames)


    def plot_peak_counts(self, binary=False, show=True):
        """

        This function requires matplotlib to be installed.

        Args:
            binary (boolean): plots 1 if a scan contained intensity, 0 otherwise.
        """
        import matplotlib.pyplot as plt
        SU = self.count_all_peaks()
        if binary:
            plt.axhline(y=self.max_scan, color='r', linestyle='-')
            plt.axhline(y=self.min_scan, color='r', linestyle='-')
        SSU = np.count_nonzero(SU, axis=1) if binary else SU.sum(axis=1)
        SSU_MS1 = SSU.copy()
        SSU_MS1[self.MS2_frameNumbers()] = 0
        SSU_MS2 = SSU.copy()
        SSU_MS2[self.MS1_frameNumbers()] = 0
        f = range(self.min_frame, self.max_frame+1)
        plt.vlines(f,0, SSU_MS1, colors='orange')
        plt.plot(f, SSU_MS2, c='grey')
        if show:
            plt.show()


class TimsDIA(AdvancedTims):
    """Data Independent Acquisition on TIMS."""
    def __init__(self, analysis_directory, use_recalibrated_state=False):
        """Construct TimsDIA.

        Basic information on frames and windows included.
        'self.frames' are indexed here both by frame and window group.

        Args:
            analysis_directory (str,Path): The '*.d' folder with the '*.tdf' files.
            use_recalibrated_state (bool): No idea yet.
        """
        super().__init__(analysis_directory, use_recalibrated_state=False)

        frame2windowGroup = self.table2df('DiaFrameMsMsInfo').set_index('Frame')
        frame2windowGroup.index.name = 'frame'
        frame2windowGroup.columns = ['window_gr']
        F = self.frames.merge(frame2windowGroup, on='frame', how='left')
        F.window_gr = F.window_gr.fillna(0).astype(int) 
        # window_gr == 0 <-> MS1 scan (quadrupole off)
        self.frames = F.reset_index().set_index(['frame', 'window_gr'])

        W = self.table2df('DiaFrameMsMsWindows')
        W['mz_left']   = W.IsolationMz - W.IsolationWidth/2.0
        W['mz_right']  = W.IsolationMz + W.IsolationWidth/2.0
        W = W.drop(columns=['IsolationMz', 'IsolationWidth', 'CollisionEnergy'])
        W.columns = 'group','scan_min','scan_max','mz_left','mz_right'
        self.min_scan = W.scan_min.min()
        self.max_scan = W.scan_max.max()
        self.border_scans = self.min_scan, self.max_scan
        MS1 = pd.DataFrame({'group':     0,
                            'scan_min':  self.min_scan,
                            'scan_max':  self.max_scan,
                            'mz_left' :  0,
                            'mz_right':  inf}, index=[0])
        W = MS1.append(W)
        W['win'] = W['window'] = range(len(W))
        windows_per_group = W.groupby('group').window.count().unique().max()
        W['stripe'] = np.where(W['window'] != 0, (W['win']-1).mod(windows_per_group) + 1, 0)
        W = W.set_index(['window', 'group'])
        W.index.names = 'window', 'window_gr'
        self.grid = sorted(list( set(W.mz_left) | set(W.mz_right) ))
        W['left'] = np.searchsorted(self.grid, W.mz_left)
        W['right'] = np.searchsorted(self.grid, W.mz_right)+1
        W['prev_left']  = [ W.left[-1]  ]*2 + list(W.left[1:-1])
        W['prev_right'] = [ W.right[-1] ]*2 + list(W.right[1:-1])
        IM_windows = W[['scan_min','scan_max']].apply(lambda x: self.scanNumToOneOverK0(1,x))
        IM_windows.columns = 'IM_min', 'IM_max'
        W = pd.concat([W, IM_windows], axis=1)
        stripes_no = (len(W)-1) // W.index.get_level_values('window_gr')[-1]
        w = np.mod(W.index.get_level_values('window').values - 1, stripes_no)
        w[0] = -1
        W['stripe'] = w
        self.windows = W

        scan_lims = W.loc[1:].copy() # data only after quadrupole selection
        intervals = pd.IntervalIndex.from_arrays(scan_lims.scan_min,
                                                 scan_lims.scan_max,
                                                 closed='both')
        intervals.name = 'scan_limits'
        scan_lims = scan_lims.reset_index()
        scan_lims.index = intervals
        scan_lims = scan_lims.set_index('window_gr', append=True)
        self.scan_lims = scan_lims[['left','right','prev_left','prev_right']]


    def plot_windows(self, query=""):
        """Plot selection windows with 'plotnine'.

        Install plotnine separately.

        Args:
           query (str): a query used for subselection in "self.windows"
        Returns:
            ggplot: a plot with selection windows
        """
        from plotnine import ggplot, aes, geom_rect, theme_minimal, xlab, ylab, labs
        D = self.windows.reset_index().query(query) if query else self.windows[1:].reset_index()
        plot = (ggplot(aes(), data=D) + 
                geom_rect(aes(xmin='mz_left', xmax='mz_right',
                              ymin='IM_min',  ymax='IM_max',
                              fill='pd.Categorical(window_gr)'), 
                          alpha=.5, color='black')+
                theme_minimal() +
                xlab('mass/charge') +
                ylab('1/K0') +
                labs(fill='Window Group'))
        return plot


    def array(self, window_grs=slice(None), 
                    frames=slice(None),
                    windows=slice(None),
                    filter_frames=''):
        """Return a numpy array with given data.

        Arguments like for 'self.get_frame_scanMin_scanMax'.

        Returns:
            np.array: an array with four columns: frame, scan, mass index (flight time), and intensity.
        """
        if window_grs == slice(None) and windows==slice(None):
            return super().array(frames=frames, filter_frames=filter_frames)
        else:
            F = self.frames.loc[(frames, window_grs),:]
            if filter_frames:
                F = F.query(filter_frames)
            F = F.index.to_frame(index=False)
            if windows != slice(None):
                W = self.windows.loc[(windows, window_grs),['scan_min', 'scan_max']]
                F = F.merge(W, on='window_gr')
                F = F.drop(columns='window_gr')
                arrays = [self.frame_array(f,s,S) for _,f,s,S in F.itertuples()]
            else:
                s, S = self.border_scans
                arrays = [self.frame_array(f,s,S) for _,f,_ in F.itertuples()]
            return np.concatenate(arrays, axis=0)


    def mzRange2windows(self, min_mz, max_mz):
        """Find numbers of windows covering the given m/z range.

        Args:
            min_mz (float): Minimal mass to charge ratio.
            min_mz (float): Maximal mass to charge ratio.
        Return:
            A set of window numbers.
        """
        pass
        return



class TimsDDA(AdvancedTims):
    """Data Dependent Acquisition on TIMS."""
    pass




