from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder  
import numpy as np


def find_best_score(X_train, y_train, start, stop, increment, learning_rates, model='XGB', skip_parameter_tuning=False):
    if skip_parameter_tuning:
        # Use default hyperparameters
        best_params = {
            'learning_rate': 0.1,
            'max_depth': 3,  # Set your default hyperparameters here
            'n_estimators': 100
        }
        best_scores = {}  # Placeholder for scores
        return best_params
    
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    # Encode the labels for both training and testing
    y_train_encoded = label_encoder.fit_transform(y_train)

    
    results = {}
    for i in range(start, stop, increment):
        #temp_params = base_parameters.copy()  # Copy base parameters
        temp_params ={'num_class':3}
        if model == 'RandomForest':
            temp_params['n_estimators'] = i
        else:  # Assuming we're tuning 'max_depth' for XGBoost in this example
            temp_params['max_depth'] = i
        
        for lr in learning_rates:
            temp_params['learning_rate'] = lr
            best_params = cross_validate_parameters(X_train, y_train_encoded, temp_params, model=model)
            results[(i, lr)] = best_params
        
        best_params,best_scores = cross_validate_parameters(X_train, y_train_encoded, temp_params, model=model)
        
        # Assuming cross_validate_hyperparameters now just evaluates a single set of parameters
        # and returns its score directly for simplification
        results[i] = best_params,best_scores
    
    #min_key = min(results, key=lambda x: results[x]['test-mlogloss-mean'])  # Find minimum based on 'test-mlogloss-mean' value
    #min_value = results[min_key]

    return results 
""" min_key, min_value, """ 


def plot_results(results):
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-')
    plt.title('Model Performance vs. Hyperparameter')
    plt.xlabel('Hyperparameter Value')
    plt.ylabel('Model Performance (Accuracy)')
    plt.grid(True)
    plt.show()

def cross_validate_parameters(X_train, y_train_encoded, parameters, model='XGB'):
    if model == 'RandomForest':
        # Define the model
        rf = RandomForestClassifier(random_state=42)
        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train_encoded)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
    else:
        # For XGBoost, we convert the data into DMatrix format which is optimized for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
        # Initialize an empty list to store evaluation scores
        eval_scores = []
        
        # XGBoost's cv function requires params to not contain lists or dicts for hyperparameters
        cv_results = xgb.cv(parameters, dtrain, num_boost_round=100, nfold=5, metrics={'mlogloss'}, early_stopping_rounds=10, as_pandas=True, seed=42)
        
        # Extract and store the evaluation scores at each boosting round
        eval_scores = cv_results['test-mlogloss-mean'].values.tolist()
        
        # Find the round with the best score
        best_round = np.argmin(eval_scores)
        
        # Retrieve the best number of boosting rounds based on the best score
        best_num_boost_round = best_round + 1  # +1 because round indexing starts from 0
        
        # Retrieve the best hyperparameters
        best_params = parameters.copy()
        best_params['num_boost_round'] = best_num_boost_round
        best_score = eval_scores[best_round]
    
    return best_params, best_score


"""
def cross_validate_parameters(X_train, y_train_encoded, parameters, model='XGB'):
    if model == 'RandomForest':
        # Define the model
        rf = RandomForestClassifier(random_state=42)
        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train_encoded)
        best_params = grid_search.best_params_
    else:
        # For XGBoost, we convert the data into DMatrix format which is optimized for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
        # XGBoost's cv function requires params to not contain lists or dicts for hyperparameters
        cv_results = xgb.cv(parameters, dtrain, nfold=5, metrics={'mlogloss'}, early_stopping_rounds=10, as_pandas=True, seed=42)
        best_num_boost_round = cv_results.shape[0]
        best_params = parameters
        best_params['num_boost_round'] = best_num_boost_round

    return best_params


"""