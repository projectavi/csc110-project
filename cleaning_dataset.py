"""
Clean datasets using Pandas.
Copyright Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""
# import pandas as pd
# import numpy as np


def remove_columns(dfs: list, cols: list[str]) -> None:
    """
    Remove the specified columns from the given dataframes.

    Precondition:
      - all(df1.columns == df2.columns for df1 in dfs for df2 in dfs)

    Instance Attributes:
      - dfs: a list containing all the dataframes that need to have columns removed.
      - cols: a list of strings containing the names of the columns that need to be removed.
    """
    for df in dfs:
        df.drop(columns=cols, inplace=True)


def add_id(dfs: list) -> None:
    """
    Add an id column to the input dataframes.

    Instance Attributes:
      - dfs: a list containing all the dataframes that need to have an 'id' column inserted.
    """
    for df in dfs:
        df.insert(0, 'id', df.index)
        df.set_index('id')


def drop_null_rows(dfs: list) -> None:
    """
    Drop all rows that are missing any values.

    Instance Attributes:
      - dfs: a list containing all the dataframes that need to have an rows with empty values
        removed.
    """
    for df in dfs:
        df.dropna(inplace=True)


def clean_dataset(dfs: list, cols: list) -> None:
    """
    Clean the input datasets using previously defined functions.

    Instance Attributes:
      - dfs: a list containing all the dataframes that need to be cleaned.
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
        'extra-imports': [],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
