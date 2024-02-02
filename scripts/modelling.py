from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def tune_model(X_train, y_train, n_estimators, max_depth=5):

    '''Tunes the model according to CV data and returns the best model and its score'''
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)

    return model, model.score(X_train, y_train)




import pandas as pd
import numpy as np

def prediction_to_csv(model):
    # Read the test data
    X_test = pd.read_csv('data/test_values.csv')

    # Select only the "age" column
    X_test_age = X_test[['age']]

    # Make predictions using the selected column
    y_pred = model.predict(X_test_age)

    # Save the predictions to a CSV file
    return np.savetxt('data/predictions.csv',X_test, y_pred, delimiter=',', fmt='%d')
