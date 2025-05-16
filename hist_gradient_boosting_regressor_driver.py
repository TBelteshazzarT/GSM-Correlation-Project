"""
 Histogram Gradient Boosting model
using processed data set with y value being storm counts
parameters from paper: 'learning_rate': 0.1,
                       'loss': 'poisson',
                       'max_bins': 50,
                       'max_depth': 200,
                       'max_features': 0.7,
                       'min_samples_leaf': 8
"""
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from Modeling_Tools import model_load
from Modeling_Tools import model_save
from Modeling_Tools import calculate_metrics

def gen_hgbr_model( x: pd.DataFrame, y: pd.DataFrame, hyperparam_tuning=False, params={}):
    """Generates the Histogram Gradient Boosting Regressor model.
    :param x: Training data x value
    :param y: Training data y value
    :param hyperparam_tuning: Boolean to enable/disable hyperparameter tuning.
     If False, model is created with the set params provided
     :param params: Dictionary providing the default params if not using hyperparameter tuning"""
    start_time = time.time()
    if hyperparam_tuning:
        param_grid = {'learning_rate': 0.1,
                      'loss': 'poisson',
                      'max_bins': 50,
                      'max_depth': 200,
                      'max_features': 0.7,
                      'min_samples_leaf': 8}
        # Hyperparameter Tuning
        model_parameter = HistGradientBoostingRegressor()
        grid = GridSearchCV(estimator=model_parameter, param_grid=param_grid, n_jobs=-1, cv=5, verbose=1)
        grid.fit(x, y)
        print(f'Hyperparameter Tuning Results: {grid.best_params_}')
        params = grid.best_params_
    model_gen = HistGradientBoostingRegressor(learning_rate=params['learning_rate'],
                                              loss=params['loss'],
                                              max_bins=params['max_bins'],
                                              max_depth=params['max_depth'],
                                              max_features=params['max_features'],
                                              min_samples_leaf=params['min_samples_leaf'],
                                              random_state=42)
    model_gen.fit(x, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Model took {elapsed_time/60} min to generate.')
    model_save(model_gen, name='HGBR.pkl')
    return model_gen

# Load model if it exists
model = model_load('HGBR.pkl')

# Load data set
data = pd.read_csv('processed_dat.csv')
df_cleaned = data.dropna()
X_labels = ['Field Magnitude Average |B|', 'f10.7_index', 'Proton Density', 'Flow Pressure',
            'Plasma (Flow) speed', 'Proton temperature', 'Na/Np', 'R', 'DST Index Min']

# Assign the data and target
X = pd.DataFrame(df_cleaned, columns=X_labels)
y = pd.Series(df_cleaned['total storms'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model is None:
    params_base = {'learning_rate': 0.1,
                   'loss': 'poisson',
                   'max_bins': 50,
                   'max_depth': 200,
                   'max_features': 0.7,
                   'min_samples_leaf': 8}
    model = gen_hgbr_model(X_train, y_train, hyperparam_tuning=False, params=params_base)

# Make Predictions
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Make Scatter Plot
ax = sns.scatterplot(x=y_test, y=y_pred)
ax = sns.scatterplot(x=y_train, y=y_pred_train)

#generate line
m = 1  # Slope
c = 0  # Intercept
x_line = np.array([0, 500])
y_line = m * x_line + c

#generate metrics
#test
r_squared_test, mse_test = calculate_metrics(y_test, y_pred)
print(f"R-squared for test: {r_squared_test}")
print(f"MSE for test: {mse_test}")
#train
r_squared_train, mse_train = calculate_metrics(y_train, y_pred_train)
print(f"R-squared for train: {r_squared_train}")
print(f"MSE for train: {mse_train}")

plt.plot(x_line, y_line, color='black') # You can change the color as needed

ax.set(xlabel="Actual Values", ylabel="Predicted Values")
plt.title("HGBR Actual vs Predicted Values")
ax.legend(['Testing', 'Training'])
plt.show()


