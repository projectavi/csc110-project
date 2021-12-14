"""
Clean datasets using Pandas.
Created by Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""
# import pandas as pd
# import numpy as np


def remove_columns(dfs: list, cols: list) -> None:
    """
    Remove the specified columns from the given dataframes.

    Precondition:
      - all(df1.columns == df2.columns for df1 in dfs for df2 in dfs)
    """
    for df in dfs:
        df.drop(columns=cols, inplace=True)


def add_id(dfs: list) -> None:
    """
    Add an id column to the input dataframes.
    """
    for df in dfs:
        df.insert(0, 'id', df.index)
        df.set_index('id')


def drop_null_rows(dfs: list) -> None:
    """
    Drop all rows that are missing any values.
    """
    for df in dfs:
        df.dropna(inplace=True)


def clean_dataset(dfs: list, cols: list) -> None:
    """
    Clean the input datasets using previously defined functions.
    """
    remove_columns(dfs, cols)
    add_id(dfs)
    drop_null_rows(dfs)


if __name__ == '__main__':
    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ["doctest", "python_ta", "datetime", "praw"],
        # the names (strs) of imported modules
        'allowed-io': ["get_comments_subreddits"],
        # the names (strs) of functions that call print/open/input
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
