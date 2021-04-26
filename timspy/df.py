import collections
import numpy as np
import pandas as pd
import pathlib
import tqdm

try:
    from fast_histogram import histogram2d
except ImportError as e:
    def histogram2d(x, y, **kwds):
        I,_,_ = np.histogram2d(x, y, **kwds)
        return I

from opentimspy.opentims import OpenTIMS, all_columns, all_columns_dtype

from .sql import tables_names, table2df
from .intensity_counts import parse_conditions_for_column_names, sum_conditioned_counters, counter2df

column2dtype = dict(zip(all_columns, all_columns_dtype))
conditions = {"singly_charged":  "inv_ion_mobility >= .0009*mz + .4744",
              "multiply_charged":"inv_ion_mobility <  .0009*mz + .4744"}


class TimsPyDF(OpenTIMS):
    """TimsData that uses info about Frames."""
    def __init__(self, analysis_directory):
        """Create an instance of the TimsPyDF.

        Args:
            analysis_directory (str, unicode string): path to the folder containing 'analysis.tdf' and 'analysis.tdf_raw'.
        """
        super().__init__(analysis_directory)
        self.frames = pd.DataFrame(self.frames)

    def min_max_measurements(self):
        """Get border values for measurements.

        Get the min-max values of the measured variables (except for TOFs, that would require iteration through data rather than parsing metadata).

        Returns:
            pd.DataFrame: Limits of individual extracted quantities. 
        """
        X = pd.DataFrame({'statistic':['min','max'],
                          'frame':[self.min_frame, self.max_frame],
                          'scan':[self.min_scan, self.max_scan],
                          'intensity':[self.min_intensity, self.max_intensity],
                          'retention_time':[self.min_retention_time, self.max_retention_time],
                          'inv_ion_mobility':[self.min_inv_ion_mobility, self.max_inv_ion_mobility],
                          'mz':[self.min_mz,self.max_mz]}).set_index('statistic')
        return X

    def table2df(self, name):
        """Retrieve a table with SQLite connection from a data base.

        Args:
            name (str): Name of the table to extract.
        Returns:
            pd.DataFrame: required data frame.
        """
        return table2df(self.analysis_directory/'analysis.tdf', name)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.peaks_cnt} peaks)"

    def summary(self):
        """Print a short summary of the data content.

        Includes the number of peaks, the minimal and the maximal frame numbers.
        """
        print(f"Peaks Inside:\t\t\t{self.peaks_cnt}")
        print(f"minimal frame:\t\t\t{self.min_frame}")
        print(f"maximal frame:\t\t\t{self.max_frame}")
        print(f"minimal scan:\t\t\t{self.min_scan}")
        print(f"maximal scan:\t\t\t{self.max_scan}")
        print(f"minimal intensity:\t\t{self.min_intensity}")
        print(f"maximal intensity:\t\t{self.max_intensity}")
        print(f"minimal retention time [s]:\t{self.min_retention_time}")
        print(f"maximal retention time [s]:\t{self.max_retention_time}")
        print(f"minimal inverse ion mobility:\t{self.min_inv_ion_mobility}")
        print(f"maximal inverse ion mobility:\t{self.max_inv_ion_mobility}")
        print(f"minimal mass to charge ratio:\t{self.min_mz}")
        print(f"maximal mass to charge ratio:\t{self.max_mz}")

    def ms1_frames_within_retention_time_limits(self, min_retention_time, max_retention_time):
        return self.frames.query("MsMsType == 0 and Time >= @min_retention_time and Time <= @max_retention_time").Id.values

    def query(self, frames, columns=all_columns):
        """Get data from a selection of frames.

        Args:
            frames (int, iterable): Frames to choose. Passing an integer results in extracting that one frame.
            columns (tuple): which columns to extract? Defaults to all possible columns.
        Returns:
            pd.DataFrame: Data frame filled with columns with raw data.
        """
        return pd.DataFrame(super().query(frames, columns))


    def plot_peak_counts(self, show=True):
        """Plot peak counts per frame.

        Arguments:
            show (bool): Show the plot immediately, or just add it to the canvas.
        """
        import matplotlib.pyplot as plt
        MS1 = self._ms1_mask
        NP = self.frames.NumPeaks
        plt.plot(self.retention_times[ MS1], NP[ MS1], label="MS1")
        plt.plot(self.retention_times[~MS1], NP[~MS1], label="MS2")
        plt.legend()
        plt.xlabel("Retention Time")
        plt.ylabel("Number of Peaks")
        plt.title("Peak Counts per Frame")
        if show:
            plt.show()


    def intensity_per_frame(self, recalibrated=True):
        """Get sum of intensity per each frame (retention time).

        Arguments:
            recalibrated (bool): Use Bruker recalibrated total intensities or calculate them from scratch with OpenTIMS?
        
        Returns:
            np.array: sums of intensities per frame. 
        """
        return self.frames.SummedIntensities.values if recalibrated else self.framesTIC()


    def iter_intensity_counts(self, frame_numbers=None, verbose=False):
        """Iterate histograms in sparse format offered by Python's Counter.

        Args:
            frame_numbers (iterable of ints): Valid frame numbers. Defaults to MS1 frames.
            verbose (boolean): Show progress bar.
        Yield:
            collections.Counter: A counter with intensities as keys and counts of these intensities as values.
        """
        if frame_numbers is None:
            frame_numbers = self.ms1_frames
        for frame_No in tqdm.tqdm(frame_numbers) if verbose else frame_numbers:
            yield collections.Counter(self.query(frame_No,["intensity"])["intensity"])


    def iter_conditional_intensity_counts(self, conditions, frame_numbers=None, verbose=False):
        """Iterate over conditional histograms, per frame basis.

        Args:
            conditions (dict): Named sets of inequalities in strings [like for pandas.DataFrame.querry].
            frame_numbers (iterable of ints): Valid frame numbers. Defaults to MS1 frames.
            verbose (boolean): Show progress bar.
        Yield:
            dict: A dictionary with keys corresponding to condition names and values being Counters.
        """
        column_names = parse_conditions_for_column_names(conditions)
        column_names.append("intensity")
        if verbose:
            print(f"Getting columns: {column_names}")
        if frame_numbers is None:
            frame_numbers = self.ms1_frames
        for frame_No in tqdm.tqdm(frame_numbers) if verbose else frame_numbers:
            frame = self.query(frame_No, column_names)
            yield {condition_name: collections.Counter(frame.query(condition).intensity) 
                   for condition_name, condition in conditions.items()}


    def intensity_distibution_df(self, conditions=conditions, frame_numbers=None, verbose=False):
        """Get the overall intensity distribution for given frames for each condition.

        Args:
            conditions (dict): Named sets of inequalities in strings [like for pandas.DataFrame.querry].
            frame_numbers (iterable of ints): Valid frame numbers. Defaults to MS1 frames.
            verbose (boolean): Show progress bar.
        Yield:
            dict: A dictionary with keys corresponding to condition names and values being Counters.
        """
        intensity_count = sum_conditioned_counters(
            self.iter_conditional_intensity_counts(conditions, frame_numbers, verbose),
            conditions)
        empty_result = all(len(cnt) == 0 for cnt in intensity_count.values())
        if empty_result:
            return pd.DataFrame()
        else:
            dfs_list = []
            for condition_name in conditions:
                df = counter2df(intensity_count[condition_name])
                df["condition_name"] = condition_name
                dfs_list.append(df)
            return pd.concat(dfs_list, ignore_index=True)


    def plot_TIC(self, recalibrated=True, show=True):
        """Plot peak counts per frame.

        Arguments:
            recalibrated (bool): Use Bruker recalibrated total intensities or calculate them from scratch with OpenTIMS?
            show (bool): Show the plot immediately, or just add it to the canvas?
        """
        import matplotlib.pyplot as plt
        MS1 = self._ms1_mask
        I = self.intensity_per_frame(recalibrated)
        plt.plot(self.retention_times[ MS1], I[ MS1], label="MS1")
        plt.plot(self.retention_times[~MS1], I[~MS1], label="MS2")
        plt.legend()
        plt.xlabel("Retention Time")
        plt.ylabel("Intensity")
        plt.title("Total Intensity [Ion Current]")
        if show:
            plt.show()


    def intensity_given_mz_inv_ion_mobility(self,
                                            frames=None,
                                            mz_bin_borders=None,
                                            inv_ion_mobility_bin_borders=None,
                                            verbose=False):
        """Sum intensity over m/z-inverse ion mobility rectangles.
    
        This function is deprecated and exists only for compatibility. Use bin_frames.

        Arguments:
            frames (iterable): Frames to consider. Defaults to all ms1_frames. 
            mz_bin_borders (np.array): Positions of bin borders for mass over charge ratios.
            inv_ion_mobility_bin_borders (np.array): Positions of bin borders for inverse ion mobilities.
        Returns:
            tuple: np.array with intensities, the positions of bin borders for mass over charge ratios and inverse ion mobilities.
        """
        if frames is None:
            frames = self.ms1_frames
        else:
            frames = list(frames)

        if mz_bin_borders is None:
            mz_bin_borders = np.arange(np.floor(self.min_mz), np.ceil(self.max_mz), 0.5) # half a dalton bins
        if inv_ion_mobility_bin_borders is None:
            inv_ion_mobility_bin_borders = np.linspace(self.min_inv_ion_mobility, self.max_inv_ion_mobility, 101) # one hundred bins in inerserve ion mobility

        frame_datasets = self.query_iter(frames=frames,
                                         columns=('mz','inv_ion_mobility','intensity'))

        I = np.zeros(shape=(len(mz_bin_borders)-1,
                            len(inv_ion_mobility_bin_borders)-1),
                     dtype=float)

        if verbose:
            frame_datasets = tqdm.tqdm(frame_datasets, total=len(frames)) 

        for X in frame_datasets:
            I_fr, _,_ = np.histogram2d(X.mz, X.inv_ion_mobility,
                                       bins=[mz_bin_borders,
                                             inv_ion_mobility_bin_borders], 
                                       weights=X.intensity)
            
            I += I_fr

        return I, mz_bin_borders, inv_ion_mobility_bin_borders

    def bin2D_frame(self,
                    frame_No,
                    min_row,
                    max_row,
                    bins_row,
                    min_column,
                    max_column,
                    bins_column,
                    variables=("mz", "inv_ion_mobility", "intensity"),
                    _save_as_uint64=True):
        """Bin a given frame into an equally-spaced grid.

        Arguments:
            frame_No (int): Number of the frame to bin.
            min_row (float): The lower border for row dimension.
            max_row (float): The upper border for row dimension.
            bins_row (int): The number of equally sized bins for row dimension.
            min_column (float): The lower border for column dimension.
            max_column (float): The upper border for column dimension.
            bins_column (int): The number of equally sized bins for column dimension.
            variables (tuple): Names of columns corresponding to rows and columns of the output. If the third value is 'intensity', output corresponds to TICs. If it is left out, the output corresponds to peak counts.
            _save_as_uint64 (bool): Recast to a more proper data format. This gives a time penalty.

        Returns:
            np.array: A 'bins_row' times 'bins_column' array with rows corresponding to 'variables[0]' and columns to 'variables[1]'. If 'variables[2]' was provided it will be used as weights, so it should be non-negative, like 'intensity'. In that case the output contains Total Ion Count. If it is not provided and 'len(variables)==2', then the output correspond to peak counts.
        """
        #TODO: move this
        frame = super().query(frame_No, columns=variables)
        x = frame[variables[0]]
        y = frame[variables[1]]
        w = frame[variables[2]] if len(variables) == 3 else None
        binned_frame = histogram2d(x, y,
                                   bins=(bins_row, bins_column),
                                   range=((min_row, max_row), (min_column, max_column)),
                                   weights=w)
        #TODO: in C++ simply use uint64_t
        return binned_frame

    def bin_frames(self,
                   frames=None,
                   variables=("mz", "inv_ion_mobility", "intensity"),
                   min_row=None,
                   max_row=None,
                   bins_row=1000,
                   min_column=None,
                   max_column=None,
                   bins_column=100,
                   desaggregate=False,
                   return_df=True,
                   multiplier=1.01,
                   verbose=False):
        """Get summary of TIC or peak count for a grid of bins in m/z and inverse ion mobility space.

        Typically it does not make too much sense to mix MS1 intensities with the others here.
        If you have fast_histogram installed, the results will be much quicker than with numpy.

        Arguments:
            frames (iterable): Frames to consider. Defaults to all ms1_frames.
            variables (tuple): Names of columns corresponding to rows and columns of the output. If the third value is 'intensity', output corresponds to TICs. If it is left out, the output corresponds to peak counts.
            min_row (float): The lower border for row dimension.
            max_row (float): The upper border for row dimension.
            bins_row (int): The number of equally sized bins for row dimension.
            min_column (float): The lower border for column dimension.
            max_column (float): The upper border for column dimension.
            bins_column (int): The number of equally sized bins for column dimension.
            desaggregate (bool): Set to True if the outcome should not be aggrageted, see 'Returns'. Watch out, you might get out of RAM if you choose too many frames or too fine a grid.
            return_df (bool): Represent the output as a pandas.DataFrame?
            multiplier (float): The relative stretch to the min and max values. New minimum is min/multiplier, new maximum is max*multiplier. This is put in place only when you do not pass in any of these values. 
            verbose (bool): Show progress bar?

        Returns:
            tuple: statistics, bin_borders. The statistic contain TICs if "variables[2] == 'instensity'", or peak counts if "variable[2]" is empty. It is an array/pd.DataFrame with size (row bins, column bins) by default, or a 3D np.array with dimensions (frame number, row bins, column bins). The 'bin_borders' consist of a dictionary of bin borders.
        """
        if frames is None:
            frames = self.ms1_frames
        else:
            frames = np.r_[frames]
        
        rows = variables[0]
        cols = variables[1]

        if min_row is None:
            min_row = getattr(self, f"min_{rows}") / multiplier

        if max_row is None:
            max_row = getattr(self, f"max_{rows}") * multiplier

        if min_column is None:
            min_column = getattr(self, f"min_{cols}") / multiplier

        if max_column is None:
            max_column = getattr(self, f"max_{cols}") * multiplier

        stats_shape = (len(frames), bins_row, bins_column) if desaggregate else (bins_row, bins_column) 
        frames_stats = np.zeros(shape=stats_shape, dtype=float)

        # We should have that fixed in C++.
        # actually, no need to zero them, but...
        bin_borders = {rows: np.linspace(min_row, max_row, bins_row+1),
                       cols: np.linspace(min_column, max_column, bins_column+1)}

        i = 0
        for frame_No in tqdm.tqdm(frames) if verbose else frames:
            frame_stats = self.bin2D_frame(frame_No,
                                           min_row,
                                           max_row,
                                           bins_row,
                                           min_column,
                                           max_column,
                                           bins_column,
                                           variables1)
            if desaggregate:
                frames_stats[i] = frame_stats
            else:
                frames_stats += frame_stats
            i += 1 # enumerate does not play well with tqdm...

        if return_df and not desaggregate:
            row_mids = (bin_borders[rows][1:] + bin_borders[rows][:-1]) / 2.0
            col_mids = (bin_borders[cols][1:] + bin_borders[cols][:-1]) / 2.0
            frames_stats_df = pd.DataFrame(frames_stats, index=row_mids, columns=col_mids)
            frames_stats_df.index.name = rows
            frames_stats_df.columns.name = cols
            frames_stats_df = frames_stats_df.astype("uint64", copy=False)
            return frames_stats_df, bin_borders
        else:
            return frames_stats, bin_borders



    def plot_intensity_given_mz_inv_ion_mobility(
            self,
            summed_intensity_matrix,
            mz_bin_borders,
            inv_ion_mobility_bin_borders,
            intensity_transformation=np.sqrt,
            interpolation='lanczos',
            aspect='auto',
            cmap='inferno',
            origin='lower',
            show=True,
            title=None,
            **kwds):
        """Sum intensity over m/z-inverse ion mobility rectangles.

        This function is deprecated and exists only for compatibility. Use bin_frames.

        Arguments:
            summed_intensity_matrix (np.array): 2D array with intensities, as produced by 'intensity_given_mz_inv_ion_mobility'.
            mz_bin_borders (np.array): Positions of bin borders for mass over charge ratios.
            inv_ion_mobility_bin_borders (np.array): Positions of bin borders for inverse ion mobilities.
            intensity_transformation (np.ufunc): Function that transforms intensities. Defaults to square root.
            interpolation (str): Type of interpolation used in 'matplotlib.pyplot.imshow'.
            aspect (str): Aspect ratio in 'matplotlib.pyplot.imshow'.
            cmap (str): Color scheme for the 'matplotlib.pyplot.imshow'.
            origin (str): Where should the origin of the coordinate system start? Defaults to bottom-left. Check 'matplotlib.pyplot.imshow'. 
            show (bool): Show the plot immediately, or just add it to the canvas?
            **kwds: Keyword arguments for 'matplotlib.pyplot.imshow' function.
        """
        import matplotlib.pyplot as plt

        plt.imshow(intensity_transformation(summed_intensity_matrix.T),
                   extent=[mz_bin_borders[0],
                           mz_bin_borders[-1],
                           inv_ion_mobility_bin_borders[0],
                           inv_ion_mobility_bin_borders[-1]],
                   interpolation=interpolation,
                   aspect=aspect,
                   cmap=cmap,
                   origin=origin,
                   **kwds)
        plt.xlabel("Mass / Charge")
        plt.ylabel("Inverse Ion Mobility")
        if title is None:
            try:
                title = f"{intensity_transformation.__name__}( Total Intensity )"
            except AttributeError:
                title = "Total Intensity"
        plt.title(title)
        if show:
            plt.show()


    def to_hdf(self,
               target_path,
               columns=all_columns,
               compression='gzip',
               compression_level=9,
               shuffle=True,
               chunks=True,
               silent=True,
               **kwds):
        """Convert the data set to HDF5 compatible with 'vaex'.

        Most of the arguments are documented on the h5py website.

        Arguments:
            target_path (str): Where to write the file (folder will be automatically created). Cannot point to an already existing file.
            columns (tuple): Names of columns to export to HDF5.
            compression (str): Compression strategy.
            compression_level (str): Parameters for compression filter.
            shuffle (bool): Enable shuffle filter.
            chunks (int): Chunk shape, or True to enable auto-chunking.
            silent (bool): Skip progress bar
        """
        import h5py

        target_path = pathlib.Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(target_path, "w") as hdf_conn:
            out_grp = hdf_conn.create_group("data")

            datasets = {}
            for colname in columns:
                datasets[colname] = out_grp.create_dataset(
                    name=colname,
                    shape=(len(self),),
                    compression=compression,
                    compression_opts=compression_level if compression=='gzip' else None,
                    dtype=column2dtype[colname],
                    chunks=chunks,
                    shuffle=shuffle,
                    **kwds)

            frame_ids = range(self.min_frame, self.max_frame+1)
            if not silent:
                frame_ids = tqdm.tqdm(frame_ids)

            data_offset = 0
            for frame_id in frame_ids:
                # super for compatibility with Michal's code...
                frame = super().query(frame_id, columns=columns)
                frame_size = len(next(frame.values().__iter__()))
                for colname, dataset in datasets.items():
                    dataset.write_direct(frame[colname], dest_sel=np.s_[data_offset:data_offset+frame_size])
                data_offset += frame_size

        if not silent:
            print(f"Finished with {target_path}")
