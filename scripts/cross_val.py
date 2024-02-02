from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score


def find_best_score(X_train, y_train, start, stop, increment, base_parameters, model='XBG'):
    results = {}
    for i in range(start, stop, increment):
        temp_params = base_parameters.copy()  # Copy base parameters
        if model == 'RandomForest':
            temp_params['n_estimators'] = i
        else:  # Assuming we're tuning 'max_depth' for XGBoost in this example
            temp_params['max_depth'] = i
        
        best_params = cross_validate_parameters(X_train, y_train, temp_params, model=model)
        
        # Assuming cross_validate_hyperparameters now just evaluates a single set of parameters
        # and returns its score directly for simplification
        results[i] = best_params['score']
    
    min_key = min(results, key=results.get)
    min_value = results[min_key]

    return min_key, min_value, results


def plot_results(results):
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-')
    plt.title('Model Performance vs. Hyperparameter')
    plt.xlabel('Hyperparameter Value')
    plt.ylabel('Model Performance (Accuracy)')
    plt.grid(True)
    plt.show()


def cross_validate_parameters(X_train, y_train, parameters, model='XBG'):
    if model == 'RandomForest':
        # Define the model
        rf = RandomForestClassifier(random_state=42)
        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
    else:
        # For XGBoost, we convert the data into DMatrix format which is optimized for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        # XGBoost's cv function requires params to not contain lists or dicts for hyperparameters
        cv_results = xgb.cv(parameters, dtrain, num_boost_round=100, nfold=5, metrics={'accuracy'}, early_stopping_rounds=10, as_pandas=True, seed=42)
        best_num_boost_round = cv_results.shape[0]
        best_params = parameters
        best_params['num_boost_round'] = best_num_boost_round

    return best_params


