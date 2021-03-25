from collections import Counter
import pandas as pd
import tqdm

from timspy.df import TimsPyDF, all_columns


def parse_conditions_for_column_names(conditions):
    used_columns = set({})
    for condition in conditions.values():
        for column_name in all_columns:
            if column_name in condition:
                used_columns.add(column_name)
    return list(used_columns)


def iter_conditional_intensity_counts(dataset, conditions, verbose=False):
    """Iterate over conditional histograms, per frame basis.

    Args:
        dataset (OpenTims): A timsTOF Pro dataset.
        verbose (boolean): Show progress bar.
    Yield:
        dict: A dictionary with keys corresponding to condition names and values being Counters.
    """
    column_names = parse_conditions_for_column_names(conditions)
    column_names.append("intensity")
    if verbose:
        print(f"Getting columns: {column_names}")
    for frame_No in tqdm.tqdm(dataset.ms1_frames) if verbose else dataset.ms1_frames:
        frame = dataset.query(frame_No, column_names)
        yield {condition_name: Counter(frame.query(condition).intensity) 
               for condition_name, condition in conditions.items()}


def sum_conditioned_counters(conditioned_counters, conditions):
    """Sum counters in dictionaries.
        
    Simply aggregates different counters.

    Args:
        conditioned_counters (iterable): Iterable of dictionaries with keys corresponding to condition names and values being Counters.
        conditions (iterable): names of all conditions.
    Returns:
        dict: A dictionary with keys corresponding to condition names and values being collections.Counter.
    """
    res = {name: Counter() for name in conditions}
    for dct in conditioned_counters:
        for name in dct:
            res[name] += dct[name]
    return res


def iter_intensity_counts(dataset, verbose=False):
    """Iterate histograms in sparse format offered by Python's Counter.

    Args:
        dataset (OpenTims): A timsTOF Pro dataset.
        verbose (boolean): Show progress bar.
    Yield:
        collections.Counter: A counter with intensities as keys and counts of these intensities as values.
    """
    for frame_No in tqdm.tqdm(dataset.ms1_frames) if verbose else dataset.ms1_frames:
        yield Counter(dataset.query(frame_No,["intensity"])["intensity"])


def sum_counters(counters):
    """Sum counters."""
    res = Counter()
    for cnt in counters:
        res += cnt
    return res


def counter2df(counter, values_name="intensity"):
    """Represent a counter as a data frame. 

    Args:
        counter (dict): A mapping between values and counts.
        values_name (str): Name for the column representing values.

    Return:
        pd.DataFrame: A data frame with values and counts, sorted by values.
    """
    return pd.DataFrame(dict(zip((values_name,"N"), zip(*counter.items())))).sort_values(values_name)