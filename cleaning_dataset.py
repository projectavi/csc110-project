"""
Clean and preprocess datasets with pandas for sentiment analysis.
Copyright Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""


def remove_columns(dfs: list, cols: list[str]) -> None:
    """
    Mutate the dataframes by removing the specified columns from the given dataframes.

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
    Mutate the dataframes by add an id column to the input dataframes.

    Instance Attributes:
      - dfs: a list containing all the dataframes that need to have an 'id' column inserted.
    """
    for df in dfs:
        df.insert(0, 'id', df.index)
        df.set_index('id')


def drop_null_rows(dfs: list) -> None:
    """
    Mutate the dataframes by dropping all rows that are missing any values.

    Instance Attributes:
      - dfs: a list containing all the dataframes that need to have an rows with empty values
        removed.
    """
    for df in dfs:
        df.dropna(inplace=True)


def clean_dataset(dfs: list, cols: list) -> None:
    """
    Clean the input dataframes by mutating them using previously defined functions.

    Instance Attributes:
      - dfs: a list containing all the dataframes that need to be cleaned.
      - cols: a list of the columns that need to be removed from each dataframe.
    """
    remove_columns(dfs, cols)
    add_id(dfs)
    drop_null_rows(dfs)


if __name__ == '__main__':
    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['pandas', 'numpy'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import pandas as pd

    dataset_filtered = pd.read_csv("https://raw.githubusercontent.com/projectavi/csc110-project"
                                   "/main/data/filtered_data.csv")

    dataset_filtered_us = pd.read_csv("https://raw.githubusercontent.com/projectavi/csc110-project"
                                      "/main/data/filtered_data_us.csv")

    dataframes = [dataset_filtered, dataset_filtered_us]
    columns = ['Unnamed: 0', 'type', 'id', 'subreddit.id', 'subreddit.name',
               'subreddit.nsfw', 'permalink', 'sentiment', 'score']

    clean_dataset(dataframes, columns)

    dataset_filtered.to_csv('dataset_filtered_pandas123.csv')
    dataset_filtered_us.to_csv('dataset_filtered_us_pandas123.csv')