from sklearn.ensemble import RandomForestClassifier

def tune_model(X_train, y_train, n_estimators, max_depth=5):

    '''Tunes the model according to CV data and returns the best model and its score'''
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)

    return model, model.score(X_train, y_train)


def prediction(model, X_test):

    return model.predict(X_test)