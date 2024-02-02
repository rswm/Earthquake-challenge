import pandas as pd
from datetime import datetime
import xgboost as xgb


def run_and_save(fitted_model, selected_test_df, building_id):
    """
    Make predictions using the selected column and save the results to a CSV file.

    Parameters:
    - fitted_model: The trained model used for making predictions.
    - selected_test_df: DataFrame containing the test features.
    - building_id: Series or a list of building IDs.

    Saves the result in two files:
    - 'data/predictions.csv' without a timestamp.
    - 'data/predictions_{timestamp}.csv' with a timestamp.
    """
    # Create a DMatrix for the test data
    d_test = xgb.DMatrix(selected_test_df)
    
    # Make predictions using the fitted model
    y_pred = fitted_model.predict(d_test)
    
    # Add +1 to each prediction
    y_pred = (y_pred + 1).astype(int)



    # Ensure building_id is a DataFrame for consistent concatenation
    if not isinstance(building_id, pd.DataFrame):
        building_id_df = pd.DataFrame(building_id, columns=['building_id'])
    else:
        building_id_df = building_id

    # Create a DataFrame with building_id and y_pred
    result_df = pd.concat([building_id_df, pd.DataFrame({'damage_grade': y_pred})], axis=1)

    # Save the combined DataFrame to a CSV file
    result_df.to_csv('data/predictions.csv', index=False)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'data/output/predictions_{timestamp}.csv'

    # Save the combined DataFrame to a CSV file with a timestamp in the filename
    result_df.to_csv(filename, index=False)

    print(f"File saved as {filename}")
