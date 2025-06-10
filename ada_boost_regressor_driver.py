"""
AdaBoost Regressor model
using processed data set with y value being DST Index
"""
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

from Modeling_Tools import model_load
from Modeling_Tools import model_save
from Modeling_Tools import calculate_metrics

def gen_abr_model( x: pd.DataFrame, y: pd.DataFrame, hyperparam_tuning=False, params={}):
    """Generates the AdaBoost Regressor model.
    :param x: Training data x value
    :param y: Training data y value
    :param hyperparam_tuning: Boolean to enable/disable hyperparameter tuning.
     If False, model is created with the set params provided
     :param params: Dictionary providing the default params if not using hyperparameter tuning"""
    start_time = time.time()
    if hyperparam_tuning:
        param_grid = {'learning_rate': 0.1,
                      'loss': 'exponential',
                      'n_estimators': 1800}
        # Hyperparameter Tuning
        model_parameter = AdaBoostRegressor()
        grid = GridSearchCV(estimator=model_parameter, param_grid=param_grid, n_jobs=-1, cv=5, verbose=1)
        grid.fit(x, y)
        print(f'Hyperparameter Tuning Results: {grid.best_params_}')
        params = grid.best_params_
    model_gen = AdaBoostRegressor(n_estimators=params['n_estimators'],
                                  loss=params['loss'],
                                  learning_rate=params['learning_rate'],
                                  random_state=42)
    model_gen.fit(x, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Model took {elapsed_time/60} min to generate.')
    model_save(model_gen, name='ada_boost_results/ABR.pkl')
    return model_gen

if __name__ == "__main__":
    # Load model if it exists
    model = model_load('ada_boost_results/ABR.pkl')

    # Load data set
    data = pd.read_csv('processed_dat.csv',  encoding='latin1')
    # Replace empty strings with NaN
    df = data.replace("", np.nan)

    # Remove rows with NaN values
    data.dropna(inplace=True)
    X_labels = ['Field Magnitude Average |B|', 'f10.7_index', 'Proton Density', 'Flow Pressure',
                'Plasma (Flow) speed', 'Proton temperature', 'Na/Np', 'R']

    # Assign the data and target
    X = pd.DataFrame(data, columns=X_labels)
    X.to_csv('data_test.csv')
    y = pd.Series(data['DST Index'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model is None:
        params_base = {'learning_rate': 0.1,
                       'loss': 'exponential',
                       'n_estimators': 1800}
        model = gen_abr_model(X_train, y_train, hyperparam_tuning=False, params=params_base)

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

    #create string for text file
    text = ''

    #generate metrics
    #test
    r_squared_test, mse_test = calculate_metrics(y_test, y_pred)
    print(f"R-squared for test: {r_squared_test}")
    text = text + f"R-squared for test: {r_squared_test}\n"
    print(f"MSE for test: {mse_test}")
    text = text + f"MSE for test: {mse_test}\n"
    #train
    r_squared_train, mse_train = calculate_metrics(y_train, y_pred_train)
    print(f"R-squared for train: {r_squared_train}")
    text = text + f"R-squared for train: {r_squared_train}\n"
    print(f"MSE for train: {mse_train}")
    text = text + f"R-squared for train: {r_squared_train}\n"

    plt.plot(x_line, y_line, color='black') # You can change the color as needed

    ax.set(xlabel="Actual Values", ylabel="Predicted Values")
    plt.title("ABR Actual vs Predicted Values")
    ax.legend(['Testing', 'Training'])
    #plt.show()
    plt.savefig('ada_boost_results/AB_actual_v_pred.png')

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for easier sorting and visualization
    feature_names = X_labels
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Print top 5 features
    print(feature_importance_df.head(5))
    text = text + f"{feature_importance_df.head(5)}\n"

    # Visualize feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(rotation=90)
    plt.tight_layout()
    #plt.show()
    plt.savefig('ada_boost_results/AB_feature_importance.png')

    file_path = "ada_boost_results/_results.txt"  # Replace with your desired file name and path

    file = open(file_path, 'w')
    file.write(text)
    file.close()