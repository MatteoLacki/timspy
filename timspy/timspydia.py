from .timspydf import TimsPyDF


class TimsPyDIA(TimsPyDF):
    def __init__(self, analysis_directory):
        """Create an instance of the TimsPyDF.

        Args:
            analysis_directory (str, unicode string): path to the folder containing 'analysis.tdf' and 'analysis.tdf_raw'.
        """
        super().__init__(analysis_directory)
        W = self.table2df('DiaFrameMsMsWindows')
        W['mz_left'] = W.IsolationMz - W.IsolationWidth/2.0
        W['mz_right'] = W.IsolationMz + W.IsolationWidth/2.0
        self.windows = W