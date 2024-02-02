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