import pandas as pd
from sklearn.preprocessing import LabelEncoder

def FeatureEngineering():
    
    #Import data

    train_values = 'data/train_values.csv'
    train_labels  = 'data/train_labels.csv'
    test_values = 'data/test_values.csv'
    

    #Load data

    tv = pd.read_csv(train_values)
    tl = pd.read_csv(train_labels)
    testv = pd.read_csv(test_values)

    #Merge data
    cdf = pd.merge(tv, tl, left_index=True, right_index=True)
    
    #Drop missing target values

    cdf  = cdf.dropna(subset=['damage_grade'])

    #Feature engineering:
    
    #Label encode geo levels

    cdf['geo_level_1_id'] = cdf['geo_level_1_id'].astype(object)
    cdf['geo_level_2_id'] = cdf['geo_level_2_id'].astype(object)
    cdf['geo_level_3_id'] = cdf['geo_level_3_id'].astype(object)

    testv['geo_level_1_id'] = testv['geo_level_1_id'].astype(object)
    testv['geo_level_2_id'] = testv['geo_level_2_id'].astype(object)
    testv['geo_level_3_id'] = testv['geo_level_3_id'].astype(object)


    return cdf

FeatureEngineering()