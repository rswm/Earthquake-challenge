from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score


def tune_model(X_train, y_train, X_val, y_val, parameters, infolist, model='XBG'):

    '''Tunes the model according to CV data and returns the best model and its score'''
    if model == 'RandomForest':

        model = RandomForestClassifier(params, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        preds_prob = model.predict(X_val)
        preds = preds_prob > 0.5
    
    else:
        params = {    
            'booster': 'gbtree',
            'objective': 'binary:logistic',
        }
        params.update(parameters)

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_val = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(params, d_train, num_boost_round=100, evals=[(d_val, "Test")], early_stopping_rounds=10)

        # Predictions
        preds_prob = model.predict(d_val)
        preds = preds_prob > 0.5

        
    infolist.append({'model': model, 'parameters': parameters, 'score': accuracy_score(y_val, preds)})
        
    return model, accuracy_score(y_val, preds)


def prediction_to_csv(model, X_test):
    # Check if the model is an instance of XGBoost
    if isinstance(model, xgb.core.Booster):
        # Convert X_test to DMatrix for XGBoost
        d_test = xgb.DMatrix(X_test)
        preds = model.predict(d_test)
    else:
        # For models like RandomForestClassifier, directly predict
        preds = model.predict(X_test)
    
    # Convert predictions array to dataframe
    preds_df = pd.DataFrame(preds, columns=['Prediction'])
    
    # Save predictions to CSV
    preds_df.to_csv('data/predictions.csv', index=False)