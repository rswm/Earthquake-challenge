import pandas as pd

def FeatureEngineering():
    
    #Import data

    train_values = 'data/train_values.csv'
    train_labels  = 'data/train_labels.csv'

    #Load data

    tv = pd.read_csv(train_values)
    tl = pd.read_csv(train_labels)
    
    #Merge data
    cdf = pd.merge(df, targetdf, left_index=True, right_index=True)
    
    #Drop missing target values

    cdf  = cdf.dropna(subset=['damage_grade'])

    #More feature engineering will be done here

    return(cdf)