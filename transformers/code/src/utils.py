import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import timedelta
import scipy


def get_df_from_excel_station_file(filename, first_data_row_i=5, col_names_row_i=3, cols_prefix="EP_station"):

    FEATURES_TARGET = [
    "EP_station_NO", "EP_station_NO2", "EP_station_NOX", "EP_station_O3",
    "EP_station_PM10", "EP_station_PM2.5", "EP_station_RH", "EP_station_SO2",
    "EP_station_Temp", "EP_station_WD", "EP_station_WS", "EP_station_Rain",
    "EP_station_CO", "EP_station_Benzene"
    ]
    

    col_names_row_i=3; cols_prefix="EP_station"; first_data_row_i=5
    all_headers = pd.read_excel(filename, header=None, skiprows=col_names_row_i, nrows=1, engine='openpyxl').iloc[0, 1:].tolist()
    cols = [f"{cols_prefix}_{c}" for c in all_headers]
    df = pd.read_excel(filename, header=None, skiprows=first_data_row_i)

    last_raw_i = df.loc[df[0] == 'ממוצע'].index.to_list()[0] # the 'ממוצע' is located at the end of the xlsx file.
    # ts_col = df[0]
    ts_col = df[0][0:last_raw_i]
    is_24_hour = ts_col.str.startswith("24:00", na=False)
    ts_col_fixed = ts_col.str.replace("24:00", "00:00")
    timestamps = pd.to_datetime(ts_col_fixed, format="%H:%M %d/%m/%Y", errors='coerce')
    timestamps.loc[is_24_hour] += pd.Timedelta(days=1)

    data_cols = df.iloc[:last_raw_i, 1:]
    values = data_cols.apply(pd.to_numeric, errors='coerce')
    values.columns = cols
    values.index = timestamps
    result_df = values

    result_df_reindexed = result_df.reindex(columns=FEATURES_TARGET)

    return result_df_reindexed

def change_index(df_to_change, df_at_timestamps):
    """
    Merges df_to_change and df_at_timestamps to create a DataFrame with all unique timestamps,
    interpolates missing values, and then returns only the values at the timestamps in df_at_timestamps.

    Parameters:
    df_to_change (pd.DataFrame): The DataFrame whose values are to be interpolated.
    df_at_timestamps (pd.DataFrame): The DataFrame with the desired timestamps.

    Returns:
    pd.DataFrame: A DataFrame with interpolated values at df_at_timestamps' timestamps.
    """
    # Merge both DataFrames by combining their indices (union of indices)
    df_merged = df_to_change.reindex(df_to_change.index.union(df_at_timestamps.index))

    # Perform linear interpolation for missing values
    df_interpolated = df_merged.interpolate(method='linear')

    # Forward fill or backward fill any remaining NaNs
    df_interpolated = df_interpolated.ffill().bfill()

    # Now, select only the rows that correspond to the index of df_at_timestamps
    df_result = df_interpolated.reindex(df_at_timestamps.index)

    return df_result

def convert_EP_ugm3_to_ppb(df_EP,df_T,df_P,cols_to_convert=None):
    # ppb = ug/m3 * R*T/P/m, df_T,df_P have both only one col, T at celcius, P at hPa
    # assuming column names e.g. EP_station_O3
    if cols_to_convert is None:
        cols_to_convert = [c for c in df_EP.columns if c.split('_')[-1] in ["NO","NO2","NOX","O3","CO","SO2"]]
    df_T_at_EP = change_index(df_T, df_EP) + 273.15 # K
    df_P_at_EP = change_index(df_P, df_EP) * 100 # Pa
    R = 8.314 # [m3 * Pa / K / mol]
    N = df_P_at_EP/R/df_T_at_EP # [mol/m3]
    for col in cols_to_convert:
        pollutant = col.split('_')[-1]
        m = None # ug/mol
        if pollutant=="NO": 
            m = 30.006*1E6
        elif pollutant=="NO2":
            m = 46.005*1E6
        elif pollutant=="NOX": 
            continue
        elif pollutant=="O3":
            m = 47.997*1E6
        elif pollutant=="CO":
            m = 28.01*1E6
        elif pollutant=="SO2":
            m = 64.06*1E6
        df_EP[f"{col}_ppb"] = df_EP[col]/m/N * 1E9
    col_NOX = [c for c in cols_to_convert if c.split('_')[-1]=="NOX"]
    col_NO = [c for c in cols_to_convert if c.split('_')[-1]=="NO"]
    col_NO2 = [c for c in cols_to_convert if c.split('_')[-1]=="NO2"]
    if len(col_NOX) == 0: return df_EP
    col_NOX=col_NOX[0]; col_NO=col_NO[0]; col_NO2=col_NO2[0]
    df_EP[f"{col_NOX}_ppb"] = df_EP[f"{col_NO}_ppb"]+df_EP[f"{col_NO2}_ppb"]
    return df_EP


def get_df_from_shamat_csv(filename, timestamp_col_name="Timestamp"):
    """
    Processes a 'shamat' CSV file containing environmental data from multiple stations, merges columns with the same
    timestamps, and translates the columns based on a provided dictionary.
    :param filename: Path to the CSV file.
    :param timestamp_col_name: The name for the timestamp column in the resulting DataFrame.
    :return: A DataFrame with merged data from all stations, with timestamps as the index and translated column names.
    """
    def get_timestamp(row):
        """Converts a timestamp string to a datetime object."""
        time_str = row[timestamp_col_name]
        return pd.to_datetime(time_str, format="%d/%m/%Y %H:%M")
    # Translation dictionary for columns
    translation_dict = {
        "לחץ בגובה התחנה (הקטופסקל)": "shamat_P",
        "קרינה מפוזרת (וואט/מ\"ר)": "beit_dagan_radiation_diffuse",  # W/m2
        "קרינה גלובלית (וואט/מ\"ר)": "beit_dagan_radiation_global",  # W/m2
        "קרינה ישירה (וואט/מ\"ר)": "beit_dagan_radiation_direct",
        "לחות יחסית (%)": "shamat_RH",
        "טמפרטורה (C°)": "shamat_T",
        "טמפרטורה ליד הקרקע (C°)": "shamat_T_ground",
        "טמפרטורת מינימום (C°)": "shamat_T_min",
        "טמפרטורת מקסימום (C°)": "shamat_T_max",
        "כיוון המשב העליון (מעלות)": "shamat_wind_gust_direction",
        "כיוון הרוח (מעלות)": "shamat_wind_direction",
        "כמות גשם (מ\"מ)": "shamat_precipitation",
        "מהירות המשב העליון (מטר לשניה)": "shamat_wind_gust_speed",
        "מהירות רוח (מטר לשניה)": "shamat_wind_speed",
        "מהירות רוח 10 דקתית מקסימלית (מטר לשניה)": "shamat_wind_speed_10min_max",
        "מהירות רוח דקתית מקסימלית (מטר לשניה)": "shamat_wind_speed_1min_max",
        "סטיית התקן של כיוון הרוח (מעלות)": "shamat_wind_direction_std_dev"
    }
    # Load the CSV file into a DataFrame, handling potential BOM and ensuring correct encoding
    df = pd.read_csv(filename, encoding='utf-8-sig')
    # Rename the second column (which usually contains the timestamp) to the consistent timestamp column name
    df.rename(columns={df.columns[1]: timestamp_col_name}, inplace=True)
    # Identify the station column (typically the first column) and drop it after separating data
    station_col = df.columns[0]
    # Convert the 'Timestamp' column to datetime
    df[timestamp_col_name] = df.apply(get_timestamp, axis=1)
    # Set the 'Timestamp' column as the index
    df.set_index(timestamp_col_name, inplace=True)
    # Separate data by station
    beit_dagan_radiation = df[df[station_col] == "בית דגן קרינה"].drop(columns=[station_col])
    beit_dagan = df[df[station_col] == "בית דגן"].drop(columns=[station_col])
    # Drop columns that contain only NaN values
    beit_dagan_radiation.dropna(axis=1, how='all', inplace=True)
    beit_dagan.dropna(axis=1, how='all', inplace=True)
    # Translate column names for each station
    beit_dagan_radiation.rename(columns=translation_dict, inplace=True)
    beit_dagan.rename(columns=translation_dict, inplace=True)
    # Merge the dataframes on the timestamp index
    df_merged = pd.merge(beit_dagan_radiation, beit_dagan, left_index=True, right_index=True, how='outer')
    # converting the shamat_wind_speed to floats instead of strings
    df_merged["shamat_wind_speed"] = pd.to_numeric(df_merged["shamat_wind_speed"], errors='coerce')
    df_merged["shamat_wind_direction"] = pd.to_numeric(df_merged["shamat_wind_direction"], errors='coerce')
    # Select only numeric columns to avoid future warnings and correctly perform mean operation
    numeric_df = df_merged.select_dtypes(include=[float, int])
    # Merge columns with the same name by averaging their values
    merged_df = numeric_df.groupby(numeric_df.columns, axis=1).mean()
    merged_df["shamat_U"] = merged_df["shamat_wind_speed"]*np.sin(np.radians(merged_df["shamat_wind_direction"])) # shamat_wind_direction is in degrees from north
    merged_df["shamat_V"] = merged_df["shamat_wind_speed"]*np.cos(np.radians(merged_df["shamat_wind_direction"])) 
    # The shamat's wind direction is from where the wind is coming! Adding new corrected columns for compitability
    merged_df["shamat_U_corrected"] = -merged_df["shamat_U"]
    merged_df["shamat_V_corrected"] = -merged_df["shamat_V"]
    return merged_df


def get_df_from_radiation_csv(filename):
    """
    Processes a 'radiation' CSV file containing radiation data.
    adapting timestamps, and translates the columns based on a provided dictionary.
    :param filename: Path to the CSV file.
    :param timestamp_col_name: The name for the timestamp column in the resulting DataFrame.
    :return: A DataFrame with merged data from all stations, with timestamps as the index and translated column names.
    """
    df = pd.read_csv(filename, encoding='utf-8', skiprows=[0], header=None)
    df = df[[1, 2]]
    col = ['timestamp', 'rad']
    df.columns = col
    df.set_index(['timestamp'], inplace=True)
    df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M")
    df['rad'] = pd.to_numeric(df['rad'], errors='coerce')
    
    return df


def station_csv_to_tensor(csv_path, timestamp_list, max_features):
    '''
    Converts a station CSV to a tensor with shape [P, L, max_features],
    padding with NaNs if the station has fewer than max_features features.

    Parameters:
        csv_path: path to CSV file with datetime index and features
        timestamp_list: list of datetime flight times
        max_features: number of features in output tensor (default=12)

    Returns:
        Tensor of shape [P, L, max_features]
    '''

    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    P = len(timestamp_list)
    L = 5 # number of lags (0, -1, -2, -3,- 4)

    # Keep only numeric columns
    df = df.select_dtypes(include=[float, int])

    # Initialize tensor with NaNs
    data_tensor = torch.full((P, L, max_features), float('nan'))

    for p, flight_time in enumerate(timestamp_list):
        for lag in range(L):
            time_lag = flight_time - pd.Timedelta(hours=lag)

            # ±7.5 minute window
            half_window = pd.Timedelta(minutes=7.5)
            window_start = time_lag - half_window
            window_end = time_lag + half_window

            mask = (df.index >= window_start) & (df.index <= window_end)
            window_df = df[mask]

            if window_df.empty:
                print(f"No data for flight_time: {flight_time}, lag: {lag}")
                continue

            # Compute mean values
            mean_vals = window_df.mean().values.astype(np.float32)

            # Pad or truncate to max_features
            if len(mean_vals) < max_features:
                padded = np.full((max_features,), np.nan, dtype=np.float32)
                padded[:len(mean_vals)] = mean_vals
            else:
                padded = mean_vals[:max_features]

            # Insert into tensor
            data_tensor[p, lag, :] = torch.from_numpy(padded)

    return data_tensor


def save_tensor_metadata(csv_path, station_name, tensor_shape, output_dir="."):
    """
    Creates and saves a JSON metadata file for a tensor.

    Parameters:
    - csv_path (str): Path to the CSV file containing feature names.
    - tensor_shape (list or tuple): Shape of the tensor, e.g., [66, 5, 12].
    - station_name (str): Name of the station (used for metadata and filename).
    - output_dir (str): Directory to save the metadata JSON file (default: current folder).

    Returns:
    - str: Path to the saved JSON file.
    """

    feature_units_dict ={
        "EP_station_NO": "ug/m3",
        "EP_station_NO2": "ug/m3",
        "EP_station_NOX": "ug/m3",
        "EP_station_O3": "ug/m3",
        "EP_station_PM10": "ug/m3",
        "EP_station_PM2.5": "ug/m3",
        "EP_station_RH": "%",
        "EP_station_SO2": "ug/m3",
        "EP_station_Temp": "C",
        "EP_station_WD": "degrees",
        "EP_station_WS": "m/s",
        "EP_station_Rain": "mm",
        "EP_station_CO": "ug/m3",
        "EP_station_Benzene": "ug/m3",
        "factor0": "",
        "shamat_P": 'Hpa',
        "shamat_RH": "%",
        "shamat_T": "C",
        "shamat_T_ground": "C",
        "shamat_T_max": "C",
        "shamat_precipitation": "mm",
        "shamat_wind_direction": "degrees",
        "shamat_wind_speed": "m/s",
        "shamat_U": "?",
        "shamat_V": "?",
        "shamat_U_corrected": "?",
        "shamat_V_corrected": "?"

    }

    # Load feature names from CSV
    feature_names = pd.read_csv(csv_path, index_col=0).columns.tolist()

    # Match units to features using the provided dictionary
    units = [feature_units_dict.get(name, "") for name in feature_names]

    # Build metadata structure
    metadata = {
        "station_name": station_name,
        "tensor_shape": list(tensor_shape),
        "features": feature_names,
        "units": units,
        "description": f"Feature metadata for station '{station_name}'"
    }

    # Define output path and save JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{station_name}.json")

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return json_path


def calculate_factor0(ep, shamat):
    NO = ep['EP_station_NO']
    rad = shamat['rad']
    temp = ep['EP_station_Temp']
    NO2 = ep['EP_station_NO2']
    F0 = np.exp(1/temp) * rad * (NO2 / NO)
    return F0
