import pandas as pd
import sqlite3


def _sql2df(path, query):
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(query, conn)


def tables_names(path):
    """List names of tables in the SQLite db.

    Arguments:
        path (str): Path to the sqlite db.

    Returns:
        pd.DataTable: table with names of tables one can get with 'table2df'.
    """
    sql = "SELECT name FROM sqlite_master WHERE TYPE = 'table'"
    return _sql2df(path, sql)


def table2df(path, name):
    """Retrieve a table with SQLite connection from a data base.

    This function is simply great for injection attacks (:
    Don't do that.

    Args:
        name (str): Name of the table to extract.
    Returns:
        pd.DataFrame: required data frame.
    """
    return _sql2df(path, f"SELECT * FROM {name}")
