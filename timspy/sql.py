import pandas as pd


def table2df(conn, name):
    """Retrieve a table with SQLite connection from a data base.

    Args:
        conn (sqlit3.Connection): A connection to an existing data base.
        name (str): Name of the table to dump into the data frame.
    Returns:
        pd.DataFrame: required data frame.
    """
    return pd.read_sql_query(f"SELECT * FROM {name}", conn)

def head(conn, name, n=10):
    return pd.read_sql_query(f"SELECT * FROM {name} LIMIT {n};", conn)    


def list_tables(conn):
    """List names of tables in the SQLite db.

    Args:
        conn (sqlite3.Connection): Connection to the data base.
    Returns:
        list: names of tables."""
    sql = "SELECT name FROM sqlite_master WHERE TYPE = 'table'"
    return [f[0] for f in conn.execute(sql)]


def get_all_tables(conn):
    """Retrieve all the tables with SQLite connection from a data base.

    Args:
        conn (sqlit3.Connection): A connection to an existing data base.

    Returns:
        dict: dictonary of data.frames
    """
    tables = list_tables(conn)
    res = {}
    for t in tables:
        try:
            res[t] = table2df(conn,t)
        except ValueError:
            print(t)
    return res
