from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator




from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def lazy_model(X_train, X_test, y_train, y_test, model=None, model_params=None):
    """
    Train a RandomForestClassifier model and calculate Mean Absolute Error (MAE).

    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training target
    - y_test: Testing target
    - model: A scikit-learn classifier model (default: RandomForestClassifier)
    - model_params: Dictionary of model hyperparameters (default: None)

    Returns:
    - Fitted model
    - MAE
    """
    if model is None:
        model = RandomForestClassifier(**(model_params or {}))

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the MAE
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae

