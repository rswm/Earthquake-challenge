from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def tune_model(X_train, y_train, n_estimators, max_depth=5):

    '''Tunes the model according to CV data and returns the best model and its score'''
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)

    return model, model.score(X_train, y_train)


def prediction_to_csv(model):

    X_test = pd.read_csv('data/test_values.csv')
    
    # Set index=False to avoid writing row numbers as a column in the csv file
    model.predict(X_test).to_csv('data/predictions.csv', index=False)   