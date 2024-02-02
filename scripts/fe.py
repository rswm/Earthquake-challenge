import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def FeatureEngineering(drop_non_numerical=False, drop_empty_rows=False):
    # Import data
    train_values = 'data/train_values.csv'
    train_labels = 'data/train_labels.csv'
    test_values = 'data/test_values.csv'

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



#Feature engineering:
    #Mean encode geo levels 1, 2 and 3
    """
    mean_encoding1 = cdf.groupby('geo_level_1_id')['damage_grade'].mean()
    mean_encoding2 = cdf.groupby('geo_level_2_id')['damage_grade'].mean()
    mean_encoding3 = cdf.groupby('geo_level_3_id')['damage_grade'].mean()

    cdf['geo_level_1_id_mean_encoded'] = cdf['geo_level_1_id'].map(mean_encoding1)
    cdf['geo_level_2_id_mean_encoded'] = cdf['geo_level_2_id'].map(mean_encoding2)
    cdf['geo_level_3_id_mean_encoded'] = cdf['geo_level_3_id'].map(mean_encoding3)
    testdf['geo_level_1_id_mean_encoded'] = testdf['geo_level_1_id'].map(mean_encoding1)
    testdf['geo_level_2_id_mean_encoded'] = testdf['geo_level_2_id'].map(mean_encoding2)
    testdf['geo_level_3_id_mean_encoded'] = testdf['geo_level_3_id'].map(mean_encoding3)
    """