"""
Best attempt on copying the results of the paper,
"Machine learning models for predicting geomagnetic
storms across five solar cycles using Dst index and heliospheric varaibles"
Daniel Boyd
"""

from Preprocessing_Tools import load_data
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
        vmin=-1,   # Set the minimum value for the color scale
        vmax=1     # Set the maximum value for the color scale
    )
    plt.title('Correlation Heatmap (Lower Triangle)')
    plt.show()

# Load the data
data = load_data('processed_dat.csv', years=(1964, 2024))
print(data.columns)
# List of columns to plot
columns_to_plot = ['total storms', 'Weak', 'Moderate', 'Strong', 'Severe', 'Extreme', 'Field Magnitude Average |B|', 'Bz GSE',
                   'Plasma (Flow) speed', 'Proton Density', 'Proton temperature', 'Na/Np', 'R', 'DST Index Min']

# Generate the time series plots
plot_time_series(data, columns_to_plot)

# Generate Correlation plot
corr_columns = ['Weak', 'Moderate', 'Strong', 'Severe', 'Extreme', 'total storms', 'Field Magnitude Average |B|',
                'Bz GSE','Field Magnitude Average |B| RMS',
                'Bz GSE RMS', 'Proton temperature', 'Proton Density', 'Plasma (Flow) speed', 'Na/Np',
                'sigma T', 'sigma N', 'sigma V', 'sigma-Na/Np', 'Kp Index Max', 'R', 'DST Index Min', 'ap-index', 'f10.7_index']
plot_correlation_heatmap(data, corr_columns)