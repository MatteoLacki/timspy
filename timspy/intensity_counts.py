from collections import Counter
import pandas as pd
import tqdm

from timspy.df import all_columns


def parse_conditions_for_column_names(conditions):
    used_columns = set({})
    for condition in conditions.values():
        for column_name in all_columns:
            if column_name in condition:
                used_columns.add(column_name)
    return list(used_columns)


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