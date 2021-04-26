import pandas as pd

from .df import TimsPyDF, conditions


def folders2intensity_distribution(folders,
                                   path_out,
                                   conditions=conditions,
                                   frame_numbers=None,
                                   verbose=True,
                                   min_retention_time=None,
                                   max_retention_time=None,
                                   _debug=False):
    intensity_counts_df = []
    for folder in folders:
        dataset = TimsPyDF( folder )
        if _debug:
            frame_numbers = list(range(1,10))
        else:
            if min_retention_time is not None and max_retention_time is not None:
                if verbose:
                    print(f"Minimal retention time: {min_retention_time}")
                    print(f"Maximal retention time: {max_retention_time}")
                frame_numbers = dataset.ms1_frames_within_retention_time_limits(min_retention_time,
                                                                                max_retention_time)
                if verbose:
                    print("Corresponding frames:")
                    print(frame_numbers)
        intensity_distr = dataset.intensity_distibution_df(conditions=conditions,
                                                           frame_numbers=frame_numbers,
                                                           verbose=verbose)
        intensity_distr["folder"] = folder
        intensity_counts_df.append( intensity_distr )
    intensity_counts_df = pd.concat( intensity_counts_df, ignore_index=True)
    intensity_counts_df.to_csv(path_out, index=False)
    if verbose:
        print("done")