import pandas as pd

from .df import TimsPyDF, conditions


def folders2intensity_distribution(folders,
                                   conditions=conditions,
                                   frame_numbers=None,
                                   verbose=True,
                                   _debug=False):
    if _debug:
        frame_numbers = list(range(1,10))
    intensity_counts_df = []
    for folder in folders:
        dataset = TimsPyDF( folder )
        intensity_distr = dataset.intensity_distibution_df(conditions=conditions,
                                                           frame_numbers=frame_numbers,
                                                           verbose=verbose)
        intensity_distr["folder"] = folder
        intensity_counts_df.append( intensity_distr )
    intensity_counts_df = pd.concat( intensity_counts_df, ignore_index=True)
    return intensity_counts_df