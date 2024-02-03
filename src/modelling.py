from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder


def tune_model(X_train, y_train, X_test, y_test, parameters, infolist, model='XGB'):
    
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    
    # Encode the labels for both training and testing
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    if model == 'RandomForest':
        model = RandomForestClassifier(**parameters, random_state=42)
        model.fit(X_train, y_train_encoded)
        preds_encoded = model.predict(X_test)
    else:
        params = {    
            'booster': 'gbtree',
            'objective': 'multi:softmax',
            'num_class': len(set(y_train_encoded))  # Dynamically set based on unique encoded labels
        }
        params.update(parameters)
        
        # Create DMatrix objects with both features and labels
        d_train = xgb.DMatrix(X_train, label=y_train_encoded)
        d_test = xgb.DMatrix(X_test, label=y_test_encoded)
        
        model = xgb.train(params, d_train, num_boost_round=100, evals=[(d_test, "Test")], early_stopping_rounds=10)
        preds_encoded = model.predict(d_test)
    
    # Decode the predictions back to the original labels
    preds_encoded = preds_encoded.astype(int)  # Ensure integer values
    preds = label_encoder.inverse_transform(preds_encoded)


    # Calculate accuracy score using the original labels
    accuracy = accuracy_score(y_test, preds)

    # Append the model details and accuracy to infolist
    infolist.append({'model': model, 'parameters': parameters, 'score': accuracy})

    return model, accuracy










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
