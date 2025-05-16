"""
Best attempt on copying the results of the paper,
"Machine learning models for predicting geomagnetic
storms across five solar cycles using Dst index and heliospheric varaibles"
Daniel Boyd
"""

import pandas as pd
from datetime import datetime, timedelta
import os
from OmniDataService import get_omni_data as get
import numpy as np
import time
from scipy.constants import mu_0, k

def load_data(name='data.csv', years=(0,2025)):
    """Loads the indicated dataset. If it can't be found it generates it new."""
    if os.path.exists(name):
        # Load base dataset
        print(f"The file '{name}' exists. Loading DataFrame...")
        data = pd.read_csv(name, index_col=0, parse_dates=True)

        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            print("Converting index to DatetimeIndex...")
            data.index = pd.to_datetime(data.index)

        return data
    else:
        # Generate the data set used in the paper
        print(f"The file '{name}' does not exist. Generating DataFrame and saving csv...")
        start_time = time.time()
        data = create_dataset(name=name, years=years)
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            print("Converting index to DatetimeIndex...")
            data.index = pd.to_datetime(data.index)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Dataset took {elapsed_time} seconds to generate.')
        return data

def create_dataset(name='data.csv', years=(0,2025)):
    """Creates a data set from the Omni low-res data that is resampled to daily and then averaged over 30day intervals"""
    # creates as csv of the DataFrame that is loaded into the variable 'data'
    start, end = years
    data = get(res='low', year_range=(start, end), flag_replace=True,  file_name=name)

    # trim data to desired fields
    #original dataset used
    #filtered_data = data[['Year', 'Decimal Day', 'Hour', 'Field Magnitude Average |B|', 'Proton temperature',
                          #'Proton Density', 'Plasma (Flow) speed', 'Na/Np', 'Flow Pressure', 'Kp', 'R',
                          #'DST Index', 'f10.7_index', 'Bz GSE']]
    filtered_data = data[['Year', 'Decimal Day', 'Hour', 'Field Magnitude Average |B|', 'Proton temperature',
                          'Proton Density', 'Plasma (Flow) speed', 'Na/Np', 'Flow Pressure', 'Kp', 'R',
                          'DST Index', 'f10.7_index', 'Bz GSE', 'sigma T', 'sigma N', 'sigma V', 'sigma-Na/Np',
                          'ap-index']]

    # convert dataset to datetime-index including hourly data
    data_conv = convert_to_datetime_index(filtered_data)

    # split out dst index to find minimum
    dst_data = data_conv[['DST Index']]
    data_conv.drop(columns=['DST Index'], inplace=True)

    #split out the kp index to find the maximum
    kp_data = data_conv[['Kp']]
    data_conv.drop(columns=['Kp'], inplace=True)

    # find minimum dst values over 30-day intervals including hourly data
    dst_min_data = find_min_over_30_day_intervals(dst_data)

    # find the maximum kp index over 30-day intervals
    kp_max_data = find_min_over_30_day_intervals(kp_data, mode='max')

    # count DST Index values within specified ranges
    ranges = [(-30, np.inf), (-50, -30), (-100, -50), (-200, -100), (-350, -200), (-np.inf, -350)]
    range_names = ["Nominal", "Weak", "Moderate", "Strong", "Severe", "Extreme"]
    dst_counts = storm_count(dst_data[['DST Index']], ranges, range_names)

    # resample the rest of the data on hour 0 and drop the 'Hour' column
    resampled_data = data_conv[data_conv['Hour'].isin([0])]
    resampled_data.drop(columns=['Hour'], inplace=True)

    #find the rms of the magnetic field magnitude and Bz
    mag_data = resampled_data[['Field Magnitude Average |B|']]
    bz_data = resampled_data[['Bz GSE']]

    mag_rms = calculate_rms_for_30day_intervals(mag_data, 'Field Magnitude Average |B|')
    bz_rms = calculate_rms_for_30day_intervals(bz_data, 'Bz GSE')

    # find averages of the rest of the values over 30-day intervals
    data_averaged = average_over_30_day_intervals(resampled_data)

    # add min dst and dst counts back to the dataframe
    data_merge = pd.concat([data_averaged, dst_min_data['DST Index'].rename("DST Index Min"), dst_counts,
                            kp_max_data['Kp'].rename('Kp Index Max'), mag_rms, bz_rms]
                           , axis=1)

    # create a total storm count
    columns_to_count = ["Weak", "Moderate", "Strong", "Severe", "Extreme"]
    data_merge['total storms'] = data_merge[columns_to_count].sum(axis=1)

    # save data to csv and return
    data_merge.to_csv(name, index=True)
    return data_merge

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
