from math import ceil
import numpy as np

from .df import TimsPyDF


class TimsPyDIA(TimsPyDF):
    def __init__(self, analysis_directory):
        """Create an instance of the TimsPyDF.

        The frames frame contains additional information about the windows.

        Args:
            analysis_directory (str, unicode string): path to the folder containing 'analysis.tdf' and 'analysis.tdf_raw'.
        """
        super().__init__(analysis_directory)
        W = self.table2df('DiaFrameMsMsWindows')
        W['mz_left'] = W.IsolationMz - W.IsolationWidth/2.0
        W['mz_right'] = W.IsolationMz + W.IsolationWidth/2.0
        self.windows = W
        
        div = ceil( len(self.frames) / (len(self.windows) + 1) )
        ms2frame2window = np.tile(np.insert(self.windows.index.values,0,-100), div) # -100 stands for MS1
        self.frames['Windows_idx'] = ms2frame2window[0:len(self.frames)]
        self.frames.Windows_idx.replace(-100, np.nan, inplace=True)


    def frames_meta(self):
        """Return meta information on every frame.

        Returns:
            pd.DataFrame: Information on every frame, including which window it belongs to.
        """
        return self.frames.merge(self.windows,
                                 left_on='Windows_idx',
                                 right_index=True,
                                 how='left')