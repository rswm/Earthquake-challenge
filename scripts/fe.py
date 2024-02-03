import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def FeatureEngineering(drop_non_numerical=False, drop_empty_rows=False):
    # Import data
    train_values = 'data/input/train_values.csv'
    train_labels = 'data/input/train_labels.csv'
    test_values = 'data/input/test_values.csv'

    # Load data
    tv = pd.read_csv(train_values)
    tl = pd.read_csv(train_labels)
    testdf = pd.read_csv(test_values)

    # Merge data
    cdf = tv.join(tl.set_index('building_id'), on='building_id')

    if drop_non_numerical:
        # Drop all non-numerical columns from the dataframes
        cdf = cdf.select_dtypes(include=[np.number])
        testdf = testdf.select_dtypes(include=[np.number])

    if drop_empty_rows:
        # Drop rows with any missing values from the dataframes
        cdf = cdf.dropna()
        testdf = testdf.dropna()

    return cdf, testdf



def mean_encode(dataframe, target_variable, columns_to_encode):
    """
    Perform mean encoding on specified columns of a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - target_variable (str): The name of the target variable column.
    - columns_to_encode (list of str): List of column names to be mean encoded.

    Returns:
    - pd.DataFrame: A DataFrame with the specified columns mean encoded.
    """
    encoded_df = dataframe.copy()
    for column in columns_to_encode:
        means = encoded_df.groupby(column)[target_variable].mean()
        encoded_df[column] = encoded_df[column].map(means)
    return encoded_df



def compute_mean_encodings(dataframe, target_variable, columns_to_encode):
    """
    Compute mean encodings for specified columns based on the target variable.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the training data.
    - target_variable (str): The name of the target variable column.
    - columns_to_encode (list of str): List of column names for which to compute mean encodings.

    Returns:
    - dict: A dictionary where keys are column names and values are Series mapping categories to mean values.
    """
    mean_encodings = {}
    for column in columns_to_encode:
        means = dataframe.groupby(column)[target_variable].mean()
        mean_encodings[column] = means
    return mean_encodings


def apply_mean_encodings(dataframe, mean_encodings):
    """
    Apply precomputed mean encodings to specified columns of a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to encode.
    - mean_encodings (dict): Precomputed mean encodings for each column.

    Returns:
    - pd.DataFrame: The DataFrame with mean encoded columns.
    """
    encoded_df = dataframe.copy()
    for column, means in mean_encodings.items():
        # Apply precomputed means and fill missing values for unseen categories
        encoded_df[column] = encoded_df[column].map(means).fillna(means.mean())
    return encoded_df
