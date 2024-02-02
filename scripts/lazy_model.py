from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator




from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

def lazy_model(X_train, X_test, y_train, y_test, model=None):
    """
    
    
    
    Train a RandomForestClassifier model and calculate Mean Absolute Error (MAE).

    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training target
    - y_test: Testing target
    - model: A scikit-learn classifier model (default: None)

    Returns:
    - Fitted model
    - MAE
    """
    rf_classifier = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0)
    
    if model is None:
        # Initialize the model with default hyperparameters
        model = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the MAE
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae

