import pandas as pd
from datetime import datetime, timedelta
import os
from OmniDataService import get_omni_data as get
import numpy as np
import time
from scipy.constants import mu_0, k

def load_data_as_pd(name):
    if os.path.exists(name):
        print(f"Loading dataset from {name}...")
        data = pd.read_csv(name, index_col=0, parse_dates=True)
        return data
    else:
        return None

def create_dataset(years, name, delay_time=24, forecast_horizon=6, feature_columns=None):
    """
    Creates a dataset from the Omni low-res data with a simulated data delay.
    If a CSV file with the given name exists, it loads the data from the CSV.
    Otherwise, it generates the dataset and saves it to the CSV.

    Parameters:
    - years: Tuple of start and end years.
    - name: Name of the file to save the dataset.
    - delay_time: Number of hours to delay dataset.
    - forecast_horizon: Number of future time steps to predict.
    - feature_columns: List of column names to use as input features.

    Returns:
    - X: Input data with shape (num_samples, num_features, sequence_length).
    - y: Target data with shape (num_samples,).
    - years_array: Array of years corresponding to each sample.
    """
    if feature_columns is None:
        feature_columns = ['DST Index']  # Default to using only 'DST Index' if no features are specified
    start_year, end_year = years
    #years_array = np.array([start_year, end_year])
    # Check if the CSV file already exists
    if os.path.exists(name):
        print(f"Loading dataset from {name}...")
        data = pd.read_csv(name, index_col=0, parse_dates=True)

        X_df = data[feature_columns]
        y_df = data['classification']
        years_array = data.index.year.to_numpy()  # Extract years from the datetime index


    else:
        # If the CSV file does not exist, generate the dataset
        print(f"Generating dataset and saving to {name}...")
        data = get(res='low', year_range=(start_year, end_year), flag_replace=True, file_name=name)
        print('Data pulled, processing...')

        # initialize booleans for saving columns
        beta = False
        temperature_save = True
        density_num_save = True
        mag_save = True

        # check if Beta is requested and what data should be added
        if 'Beta' in feature_columns:
            beta = True
            feature_columns.remove('Beta')
            if 'Proton temperature' not in feature_columns:
                feature_columns.append('Proton temperature')
                temperature_save = False

            if 'Proton Density' not in feature_columns:
                feature_columns.append('Proton Density')
                density_num_save = False

            if 'Field Magnitude Average |B|' not in feature_columns:
                feature_columns.append('Field Magnitude Average |B|')
                mag_save = False

        # Include specified columns for multivariate input
        filtered_data = data[['Year', 'Decimal Day', 'Hour'] + feature_columns].copy()
        data_conv = convert_to_datetime_index(filtered_data)
        df_cleaned = data_conv.dropna()
        df_cleaned['classification'] = df_cleaned['DST Index'].apply(classify_dst)
        if beta:
            # generate beta column
            df_cleaned['Beta'] = df_cleaned.apply(calc_beta, axis=1)
            feature_columns.append('Beta')

            # initialize columns as not included
            feature_columns.remove('Proton temperature')
            feature_columns.remove('Proton Density')
            feature_columns.remove('Field Magnitude Average |B|')
            if temperature_save:
                feature_columns.append('Proton temperature')
            if density_num_save:
                feature_columns.append('Proton Density')
            if mag_save:
                feature_columns.append('Field Magnitude Average |B|')

        # Use specified columns for X
        X_df = df_cleaned[feature_columns] # filters out y
        # Shift y data to account for delay and forecast
        y_df = df_cleaned['classification'].shift(delay_time + forecast_horizon)
        years_array = df_cleaned.index.year.to_numpy()  # Extract years from the datetime index


        # Combine X, y, and years into one dataframe
        data_to_save = pd.concat([X_df, y_df], axis=1)

        # Resample to hourly
        data_resampled = data_to_save.resample('D').mean()
        # Remove fill values from shifting
        data_resampled.dropna(inplace=True)

        # Save dataframe
        data_resampled.to_csv(name, index=True, encoding='latin1')
        X_df = data_resampled[feature_columns]
        y_df = data_resampled['classification']

    # Convert Dataframes to NumPy arrays
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    X.transpose()

    return X, y, years_array

def average_over_30_day_intervals(df):
    """
    Averages all the values in each column of the DataFrame over 30-day intervals.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a DateTime index and daily values.

    Returns:
    pd.DataFrame: A new DataFrame with the averaged values over 30-day intervals.
    """

    # Ensure the DataFrame has a DateTime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame must have a DateTime index.")

    # Resample the data by 30-day intervals and calculate the mean
    resampled_df = df.resample('30D').mean()

    return resampled_df

def convert_to_datetime_index(df):
    """
    Converts a DataFrame with 'Year' and 'Decimal Day' columns into a DataFrame with a datetime index.

    Parameters:
    df (pd.DataFrame): Input DataFrame with 'Year' and 'Decimal Day' columns.

    Returns:
    pd.DataFrame: A new DataFrame with a datetime index.
    """

    # Ensure the required columns exist
    if 'Year' not in df.columns or 'Decimal Day' not in df.columns:
        raise ValueError("The DataFrame must contain 'Year' and 'Decimal Day' columns.")

    # Create a datetime index
    datetime_index = []
    for year, decimal_day, hour in zip(df['Year'], df['Decimal Day'], df['Hour']):
        # Calculate the integer day of the year and the fractional part
        day_of_year = int(decimal_day)
        fractional_day = decimal_day - day_of_year

        # Create a datetime object for the start of the year
        start_of_year = datetime(year=year, month=1, day=1)

        # Add the days, fractional days, and hours to the start of the year
        dt = start_of_year + timedelta(days=day_of_year - 1, seconds=int(fractional_day * 86400)) + timedelta(hours=hour)
        datetime_index.append(dt)

    # Create a new DataFrame with the datetime index
    df_with_datetime = df.copy()
    df_with_datetime.index = pd.to_datetime(datetime_index)

    # Drop the 'Year' and 'Decimal Day' columns as they are no longer needed
    df_with_datetime = df_with_datetime.drop(columns=['Year', 'Decimal Day'])

    return df_with_datetime

def find_min_over_30_day_intervals(df, mode = 'min'):
    """
    Finds the minimum of all the values in each column of the DataFrame over 30-day intervals.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a DateTime index and daily values.
    mode (str): Enter 'min' or 'max' depending on desired operations

    Returns:
    pd.DataFrame: A new DataFrame with the minimum values over 30-day intervals.
    """

    # Ensure the DataFrame has a DateTime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame must have a DateTime index.")

    # Resample the data by 30-day intervals and calculate the minimum
    if mode == 'min':
        resampled_df = df.resample('30D').min()
    elif mode == 'max':
        resampled_df = df.resample('30D').max()
    else:
        print(f'Invalid mode selection: {mode}')
    return resampled_df

def storm_count(df, ranges, column_names, interval_days=30):
    """
    Helper function to count daily minimum values within specified ranges for each 30-day interval.
    Supports open bounds (e.g., -infinity to 0 or 0 to infinity).

    Parameters:
        df (pd.DataFrame): Input DataFrame with datetime index and hourly data.
        ranges (list of tuples): List of ranges to compare values against (e.g., [(-np.inf, 0), (0, np.inf)]).
        column_names (list of str): Names for the output columns corresponding to each range.
        interval_days (int): Number of days in each interval (default is 30).

    Returns:
        pd.DataFrame: A DataFrame with counts of days falling into each range for each interval.
    """
    #TODO modify the function to return counts on the last row (partial intervals)

    # Ensure the input DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a datetime index.")

    # Resample the DataFrame to get the daily minimum values
    #daily_min = df.resample('D').min()

    # leave it hourly for higher counts (keeping variable name for easy editing)
    daily_min = df.copy()

    # Initialize a dictionary to store counts for each range
    range_counts = {name: [] for name in column_names}

    # Initialize a list to store the start dates of each interval
    interval_starts = []

    # Iterate over the DataFrame in intervals of `interval_days`
    start_date = daily_min.index[0]
    while start_date < daily_min.index[-1]:
        # Define the end date of the interval
        end_date = start_date + pd.Timedelta(days=interval_days)

        # Filter the daily minimum values for the current interval
        interval_data = daily_min.loc[start_date:end_date]

        # Count the number of days falling into each range
        for (range_start, range_end), col_name in zip(ranges, column_names):
            # Handle open bounds
            if range_start == -np.inf:
                count = (interval_data < range_end).sum().values[0]  # Extract the count value
            elif range_end == np.inf:
                count = (interval_data >= range_start).sum().values[0]  # Extract the count value
            else:
                count = ((interval_data >= range_start) & (interval_data < range_end)).sum().values[0]  # Extract the count value
            range_counts[col_name].append(count)

        # Append the start date of the interval
        interval_starts.append(start_date)

        # Move to the next interval
        start_date = end_date

    # Create the output DataFrame
    result_df = pd.DataFrame(range_counts, index=interval_starts)
    result_df.index.name = 'Interval Start'

    return result_df


def calculate_rms_for_30day_intervals(df, column_name):
    """
    Calculate the Root Mean Square (RMS) for each 30-day interval in a pandas DataFrame column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the daily data.
    column_name (str): The name of the column to calculate the RMS for.

    Returns:
    pd.DataFrame: A DataFrame with the RMS values for each 30-day interval, indexed by the start date of each interval.
    """

    # Ensure the DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Create a copy of the data and initialize lists to store RMS values and interval start dates
    df_input = df.copy()
    rms_values = []
    interval_starts = []

    # Group the data into 30-day intervals
    for interval_start, group in df_input.resample('30D'):
        if not group.empty:
            # Calculate the RMS for the current 30-day interval
            rms = np.sqrt((group[column_name] ** 2).mean())
            rms_values.append(rms)
            interval_starts.append(interval_start)

    # Create a DataFrame with the RMS values and datetime index
    name = column_name + ' RMS'
    rms_df = pd.DataFrame(rms_values, columns=[name], index=interval_starts)
    #rms_df.index.name = 'Interval Start'

    return rms_df

def classify_dst(x:float):
    """Function that operates inside a .apply() call. Creates a new column with discrete labels for storm category"""
    match x:
        # Quiet
        case _ if x > -30:
            return 0
        # Small Storm
        case _ if -50 < x <= -30:
            return 1
        # Moderate Storm
        case _ if -80 < x <= -50:
            return 2
        # Severe Storm
        case _ if -100 < x <= -80:
            return 3
        # Extreme Storm
        case _ if x <= -100:
            return 4

def calc_beta(row):
    """Function that operates inside a .apply() call. Creates a new column with Beta calculated from temp,
     density number and magnetic field"""
    t = row['Proton temperature']
    np = row['Proton Density']
    b = row['Field Magnitude Average |B|']
    beta = (np * k * t)/((b * b) * mu_0)
    return beta