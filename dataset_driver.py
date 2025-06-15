"""
Best attempt on copying the results of the paper,
"Machine learning models for predicting geomagnetic
storms across five solar cycles using Dst index and heliospheric varaibles"
Daniel Boyd
"""

from Preprocessing_Tools import create_dataset, load_data_as_pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def plot_time_series(data, columns):
    """
    Plots time series for a list of columns in the DataFrame as separate figures.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data with a DateTime index.
    columns (list of str): List of column names to plot.
    """
    # Plot each column in a separate figure
    for column in columns:
        plt.figure(figsize=(10, 6))  # Create a new figure for each column
        plt.plot(data.index, data[column])
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.title(f'{column}')
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()  # Display the current figure


def plot_correlation_heatmap(data, columns):
    """
    Creates a correlation table and visualizes it as a heatmap using Seaborn.
    A mask is applied to remove redundant values (upper triangle of the correlation matrix).

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    columns (list of str): List of column names to include in the correlation analysis.
    """
    # Subset the data to include only the specified columns
    subset_data = data[columns]

    # Calculate the correlation matrix
    correlation_matrix = subset_data.corr()

    # Create a mask to hide the upper triangle of the heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        #cmap='Reds',
        fmt='.2f',
        linewidths=0.5,
        mask=mask,  # Apply the mask to hide the upper triangle
        vmin=-1,  # Set the minimum value for the color scale
        vmax=1  # Set the maximum value for the color scale
    )
    plt.title('Correlation Heatmap (Lower Triangle)')
    plt.show()

if __name__ == "__main__":
    features = [    'Field Magnitude Average |B|',
                'Bz GSE', 'Proton temperature', 'Proton Density', 'Plasma (Flow) speed', 'Na/Np',
                'sigma T', 'sigma N', 'sigma V', 'sigma-Na/Np', 'Kp', 'R', 'DST Index', 'ap-index',
                'f10.7_index']

    # Load the data
    data = create_dataset(name='processed_dat.csv', years=(1964, 2024), delay_time=0, forecast_horizon=0, feature_columns=features)
    data = load_data_as_pd(name='processed_dat.csv')
    print(data.columns)

    # Generate the time series plots
    plot_time_series(data, features)

    # Generate Correlation plot

    plot_correlation_heatmap(data, features)
