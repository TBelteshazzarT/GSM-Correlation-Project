"""
General Modeling Tools

"""
import os
import pickle
import numpy as np


def model_load(name='model.pkl'):
    """Loads the indicated model. If it can't be found it generates it None."""
    if os.path.exists(name):
        print(f"The file '{name}' exists. Loading model...")
        file_name, file_extension = os.path.splitext(name)
        if file_extension == '.pkl':
            # Load base model
            with open(name, 'rb') as f:
                model_to_load = pickle.load(f)
        else:
            print(f'Improper filetype: {file_extension}')
            model_to_load = None
        return model_to_load
    else:
        print(f"The file '{name}' was not found. Generating model (this may take a while)...")
        return None

def model_save(model_to_save, name='model.pkl'):
    """Saves the indicated model."""
    with open(name, 'wb') as f:
        pickle.dump(model_to_save, f)
    print(f'Model saved as {name}')



def calculate_metrics(y_true, y_pred):
    """
    Calculate R-squared and Mean Squared Error (MSE) for given true and predicted values.

    Parameters:
    y_true (numpy.ndarray): The true values.
    y_pred (numpy.ndarray): The predicted values.

    Returns:
    r_squared (float): The R-squared value.
    mse (float): The Mean Squared Error.
    """

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)

    # Calculate the total sum of squares (TSS)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)

    # Calculate the residual sum of squares (RSS)
    rss = np.sum((y_true - y_pred) ** 2)

    # Calculate R-squared
    r_squared = 1 - (rss / tss)

    return r_squared, mse