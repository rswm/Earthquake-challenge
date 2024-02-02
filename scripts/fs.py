import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def FeatureSelection(dataframe, selected_features=None):
    """
    Perform feature selection on a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing features.
        selected_features (list, optional): List of feature names to select. Default is None, which selects all features.

    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    if selected_features is None:
        # If no selected_features provided, return the original DataFrame
        return dataframe
    else:
        # Filter the DataFrame to include only selected features
        selected_df = dataframe[selected_features]
        return selected_df


def train_test_split_function(dataframe, target_column_name, test_size=0.2, random_state=None):
    """
    Split a DataFrame into X_train, X_test, y_train, and y_test.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing features and target column.
        target_column_name (str): The name of the target column.
        test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int or None, optional): Seed for the random number generator. Default is None.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test DataFrames.
    """
    # Separate features (X) and target (y)
    
    X = dataframe.drop(columns=[target_column_name])
    y = dataframe[target_column_name]
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test






def check_numerical_columns(df):
    # Check if all columns are numeric by using select_dtypes.
    non_numerical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # If the list of non-numerical columns is empty, print "Yes".
    if len(non_numerical_columns) == 0:
        print("Yes")
    else:
        # Otherwise, print the list of non-numerical column names.
        print(non_numerical_columns)


