"""
IGRA (Integrated Global Radiosonde Archive) Toolkit

This library provides tools for working with IGRA radiosonde data, including:
- Reading and parsing station metadata
- Processing individual station data files
- Data analysis and visualization
- Data export and conversion
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4 as nc
import os
import requests
import zipfile
from urllib.parse import urljoin
import datetime
import io
import shutil
import argparse
from tqdm import tqdm
import igratmetadata as metadata
import plotly.express as px
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import mplcursors
from matplotlib.widgets import Button
import json

def download_station_file(station_id: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Download and unzip a station's data file from the IGRA archive.
    
    Args:
        station_id: Station ID (e.g., "USM00072520" for Albany, NY)
        output_dir: Optional directory to save the unzipped file. If None, uses current directory.
        
    Returns:
        Path to the unzipped file if successful, None otherwise.
        
    Examples:
        >>> # Download data for a single station
        >>> station = "USM00072520"  # Albany, NY station
        >>> file_path = download_station_file(station)
        >>> print(f"Downloaded file: {file_path}")
    """
    try:
        if output_dir is None:
            output_dir = os.getcwd()
            
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the URL for the station's zip file
        base_url = metadata.IGRA_FILES_URL
        zip_filename = f"{station_id}-data.txt.zip"
        zip_url = urljoin(base_url, zip_filename)
        
        # Download the zip file directly to memory
        print(f"Downloading data for station {station_id}...")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Create a BytesIO object to hold the zip file in memory
        zip_buffer = io.BytesIO()
        
        # Download with progress bar
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {station_id}") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    zip_buffer.write(chunk)
                    pbar.update(len(chunk))
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Extract the text file
        print(f"Extracting data for station {station_id}...")
        with zipfile.ZipFile(zip_buffer) as zip_ref:
            # Get the name of the text file in the zip
            txt_file_name = None
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.txt'):
                    txt_file_name = file_info.filename
                    break
            
            if txt_file_name:
                # Extract the text file to the output directory
                output_path = os.path.join(output_dir, txt_file_name)
                with zip_ref.open(txt_file_name) as txt_file, open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(txt_file, out_file)
                print(f"Successfully downloaded and extracted data to: {output_path}")
                return output_path
            else:
                raise Exception("No text file found in the zip archive")
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data for station {station_id}: {e}")
        return None
    except zipfile.BadZipFile:
        print(f"Error: The downloaded file for {station_id} is not a valid zip file.")
        return None
    except Exception as e:
        print(f"Unexpected error processing data for {station_id}: {e}")
        return None
    
def read_station_data(station_id: str, 
                      path: Optional[str] = None,
                      main: bool = True,
                      download_dir: Optional[str] = None,
                     start_date: Optional[datetime.datetime] = None,
                     end_date: Optional[datetime.datetime] = None, 
                     file_type: Optional[str] = 'netcdf') -> Union[pd.DataFrame, xr.Dataset]:
    """
    Read and parse a single station's IGRA data file.
    
    Args:
        station_id: Station ID (e.g., "USM00072520" for Albany, NY)
        path: Optional path to the station's data file
        main: Whether to include only main variables (True) or all variables (False)
        download_dir: Directory to save the output file 
        start_date: Optional start date to filter data
        end_date: Optional end date to filter data
        file_type: Optional file type to return, either 'netcdf', 'pandas', or 'df'
        
    Returns:
        If file_type is 'netcdf':
            xarray.Dataset containing sounding data with dimensions:
            - num_profiles: Number of soundings
            - levels: Number of levels in each sounding
            - variables: Pressure, height, temperature, dewpoint, wind_direction, wind_speed, relative_humidity
            Data is saved as a netcdf file. 
            
        If file_type is 'pandas' or 'df':
            DataFrame containing sounding data with columns:
            - datetime: Timestamp of the sounding
            - pressure: Pressure in hPa
            - height: Height in meters
            - temperature: Temperature in Celsius
            - dewpoint: Dewpoint temperature in Celsius
            - wind_direction: Wind direction in degrees
            - wind_speed: Wind speed in m/s
            - relative_humidity: Relative humidity in percent
            - level_type1: Level type 1 (if main=False)
            - level_type2: Level type 2 (if main=False)
            - elapsed_time: Elapsed time (if main=False)
            - pressure_flag: Pressure flag (if main=False)
            - height_flag: Height flag (if main=False)
            - temperature_flag: Temperature flag (if main=False)
            Data is saved as a csv file. 

    Examples:
        >>> # Read data for a single station and return as NetCDF
        >>> station = "USM00072520"  # Albany, NY station
        >>> ds = read_station_data(station, file_type='netcdf')
        >>> print(ds.dims)
        {'num_profiles': 1000, 'levels': 100}
        
        >>> # Read data and return as DataFrame
        >>> df = read_station_data(station, file_type='df')
        >>> print(df.columns)
        ['datetime', 'pressure', 'height', 'temperature', 'relative_humidity', 
         'wind_direction', 'wind_speed', 'dewpoint_depression']
        
        >>> # Read data with date filtering and all variables
        >>> start = datetime.datetime(2020, 1, 1)
        >>> end = datetime.datetime(2020, 1, 31)
        >>> df = read_station_data(station, 
        ...                       start_date=start,
        ...                       end_date=end,
        ...                       main=False,
        ...                       file_type='df')
        >>> print(df.columns)
        ['datetime', 'pressure', 'height', 'temperature', 'relative_humidity',
         'wind_direction', 'wind_speed', 'dewpoint_depression', 'level_type1',
         'level_type2', 'elapsed_time', 'pressure_flag', 'height_flag',
         'temperature_flag']
    """

    try:
        # Set up the output file path
        if download_dir is None:
            download_dir = os.getcwd()
        elif not os.path.exists(download_dir):
            os.makedirs(download_dir)

        if file_type.lower() in ['netcdf', 'nc']:
            if main:
                output_file = os.path.join(download_dir, f"{station_id}-main.nc")
            else:
                output_file = os.path.join(download_dir, f"{station_id}-full.nc")
        else:  # pandas/df
            if main:
                output_file = os.path.join(download_dir, f"{station_id}-main.csv")
            else:
                output_file = os.path.join(download_dir, f"{station_id}-full.csv")

        if path is None:
            base_url = metadata.IGRA_FILES_URL

            # Construct the URL for the station's zip file
            zip_filename = f"{station_id}-data.txt.zip"
            zip_url = urljoin(base_url, zip_filename)
            
            try:
                # Download the zip file directly to memory
                print(f"Downloading data for station {station_id}...")
                response = requests.get(zip_url, stream=True)
                response.raise_for_status()
                
                # Get the total file size
                total_size = int(response.headers.get('content-length', 0))
                
                # Create a BytesIO object to hold the zip file in memory
                zip_buffer = io.BytesIO()
                
                # Download with progress bar
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {station_id}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            zip_buffer.write(chunk)
                            pbar.update(len(chunk))
                
                # Reset buffer position
                zip_buffer.seek(0)
                
                # Extract the text file directly to memory
                print(f"Extracting data for station {station_id}...")
                with zipfile.ZipFile(zip_buffer) as zip_ref:
                    # Get the name of the text file in the zip
                    txt_file_name = None
                    for file_info in zip_ref.infolist():
                        if file_info.filename.endswith('.txt'):
                            txt_file_name = file_info.filename
                            break
                    
                    if txt_file_name:
                        # Read the text file directly into memory
                        with zip_ref.open(txt_file_name) as txt_file:
                            data_content = txt_file.read().decode('utf-8')
                    else:
                        raise Exception("No text file found in the zip archive")
                
                print(f"Successfully downloaded and extracted data for {station_id}")
            
            except requests.exceptions.RequestException as e:
                print(f"Error downloading data for station {station_id}: {e}")
                return None
            except zipfile.BadZipFile:
                print(f"Error: The downloaded file for {station_id} is not a valid zip file.")
                return None
            except Exception as e:
                print(f"Unexpected error processing data for {station_id}: {e}")
                return None
        else:
            # Read from local file
            with open(path, 'r') as f:
                data_content = f.read()

        # Process the data from memory
        lines = data_content.splitlines()
        
        # First pass: count soundings and find max levels
        sounding_count = 0
        max_levels = 0
        current_levels = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                sounding_count += 1
                if current_levels > max_levels:
                    max_levels = current_levels
                current_levels = 0
            else:
                current_levels += 1
        
        # Check the last sounding
        if current_levels > max_levels:
            max_levels = current_levels
        
        print(f"Found {sounding_count} soundings with maximum {max_levels} levels")
        
        # Create arrays to store data
        dates = np.zeros(sounding_count, dtype=np.int32)
        times = np.zeros(sounding_count, dtype=np.int32)
        reltimes = []
        numlevs = np.zeros(sounding_count, dtype=np.int32)
        p_srcs = []
        np_srcs = []

        pressure = np.full((sounding_count, max_levels), np.nan)
        gph = np.full((sounding_count, max_levels), np.nan)
        temp = np.full((sounding_count, max_levels), np.nan)
        rh = np.full((sounding_count, max_levels), np.nan)
        wspd = np.full((sounding_count, max_levels), np.nan)
        wdir = np.full((sounding_count, max_levels), np.nan)
        lvltyp1 = np.full((sounding_count, max_levels), np.nan)
        lvltyp2 = np.full((sounding_count, max_levels), np.nan)
        etime = np.full((sounding_count, max_levels), np.nan)
        pflag = np.full((sounding_count, max_levels), np.nan)
        zflag = np.full((sounding_count, max_levels), np.nan)
        tflag = np.full((sounding_count, max_levels), np.nan)
        dpdp = np.full((sounding_count, max_levels), np.nan)
        
        # Second pass: read data
        sounding_idx = -1
        level_idx = 0
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('#'):
                sounding_idx += 1
                level_idx = 0
                
                # Parse header
                year = int(line[13:17])
                month = int(line[18:20])
                day = int(line[21:23])
                hour = int(line[24:26])
                reltime = line[27:31].strip()
                numlev = int(line[32:36])
                p_src = line[37:45].strip()
                np_src = line[46:54].strip()
                
                # Store date as YYYYMMDD
                dates[sounding_idx] = year * 10000 + month * 100 + day
                times[sounding_idx] = hour
                numlevs[sounding_idx] = numlev
                
                # Ensure the entire string is assigned, not just the first character
                if reltime != '':
                    reltimes.append(reltime)
                else:
                    reltimes.append('--')
                if p_src != '':
                    p_srcs.append(p_src)
                else:
                    p_srcs.append('--')

                if np_src != '':
                    np_srcs.append(np_src)
                else:
                    np_srcs.append('--')
                
            else:
                # Parse data line
                try:
                    # IGRA data format: columns are fixed width
                    lvltyp1_val = line[0:1]
                    lvltyp2_val = line[1:2]
                    etime_val = line[3:8].strip()
                    press_val = line[9:15].strip()
                    pflag_val = line[15:16]
                    gph_val = line[16:21].strip()
                    zflag_val = line[21:22]
                    temp_val = line[22:27].strip()
                    tflag_val = line[27:28]
                    rh_val = line[28:33].strip()
                    dpdp_val = line[34:39].strip()
                    wdir_val = line[40:45].strip()
                    wspd_val = line[46:51].strip()
                    
                    # Convert values to float if not empty, otherwise keep as NaN
                    if press_val and press_val != '-9999':
                        pressure[sounding_idx, level_idx] = float(press_val)/100
                    elif press_val:
                        pressure[sounding_idx, level_idx] = float(press_val)
                    
                    if gph_val and gph_val != '-9999' and gph_val != '-8888':
                        gph[sounding_idx, level_idx] = float(gph_val)
                        
                    if temp_val and temp_val != '-9999' and temp_val != '-8888':
                        temp[sounding_idx, level_idx] = float(temp_val)/10
                    elif temp_val:
                        temp[sounding_idx, level_idx] = float(temp_val)
                        
                    if rh_val and rh_val != '-9999' and rh_val != '-8888':
                        rh[sounding_idx, level_idx] = float(rh_val)/10
                    elif rh_val:
                        rh[sounding_idx, level_idx] = float(rh_val)
                        
                    if wdir_val and wdir_val != '-9999' and wdir_val != '-8888':
                        wdir[sounding_idx, level_idx] = float(wdir_val)
                        
                    if wspd_val and wspd_val != '-9999' and wspd_val != '-8888':
                        wspd[sounding_idx, level_idx] = float(wspd_val)/10
                    elif wspd_val:
                        wspd[sounding_idx, level_idx] = float(wspd_val)
                    
                    # Store level type and flags as numeric values if possible
                    if lvltyp1_val and lvltyp1_val.isdigit():
                        lvltyp1[sounding_idx, level_idx] = float(lvltyp1_val)
                        
                    if lvltyp2_val and lvltyp2_val.isdigit():
                        lvltyp2[sounding_idx, level_idx] = float(lvltyp2_val)
                        
                    if etime_val:
                        etime[sounding_idx, level_idx] = float(etime_val)
                        
                    if pflag_val and pflag_val.isalnum():
                        # Store as ASCII value if it's a character
                        pflag[sounding_idx, level_idx] = ord(pflag_val) if len(pflag_val) == 1 else np.nan
                        
                    if zflag_val and zflag_val.isalnum():
                        zflag[sounding_idx, level_idx] = ord(zflag_val) if len(zflag_val) == 1 else np.nan
                        
                    if tflag_val and tflag_val.isalnum():
                        tflag[sounding_idx, level_idx] = ord(tflag_val) if len(tflag_val) == 1 else np.nan
                        
                    if dpdp_val and dpdp_val != '-9999' and dpdp_val != '-8888':
                        dpdp[sounding_idx, level_idx] = float(dpdp_val)/10
                    elif dpdp_val:
                        dpdp[sounding_idx, level_idx] = float(dpdp_val)

                    
                    level_idx += 1
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error details: {e}")
                    # Skip malformed lines
                    continue

        # Create the NetCDF file
        if file_type.lower() in ['netcdf', 'nc']:
            with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
                # Create dimensions
                ncfile.createDimension('num_profiles', sounding_count)
                ncfile.createDimension('levels', max_levels)
                
                # Create variables
                if main:
                    date_var = ncfile.createVariable('date', 'i4', ('num_profiles',))
                    time_var = ncfile.createVariable('time', 'i4', ('num_profiles',))
                    pressure_var = ncfile.createVariable('pressure', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    gph_var = ncfile.createVariable('gph', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    temperature_var = ncfile.createVariable('temperature', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    rh_var = ncfile.createVariable('rh', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    wdir_var = ncfile.createVariable('wdir', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    wspd_var = ncfile.createVariable('wspd', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    dpdp_var = ncfile.createVariable('dpdp', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                else:
                    date_var = ncfile.createVariable('date', 'i4', ('num_profiles',))
                    time_var = ncfile.createVariable('time', 'i4', ('num_profiles',))
                    reltime_var = ncfile.createVariable('reltime', str, ('num_profiles',))
                    numlev_var = ncfile.createVariable('numlev', 'i4', ('num_profiles',))
                    p_src_var = ncfile.createVariable('p_src', str, ('num_profiles',))
                    np_src_var = ncfile.createVariable('np_src', str, ('num_profiles',))
                    pressure_var = ncfile.createVariable('pressure', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    gph_var = ncfile.createVariable('gph', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    temperature_var = ncfile.createVariable('temperature', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    rh_var = ncfile.createVariable('rh', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    wdir_var = ncfile.createVariable('wdir', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    wspd_var = ncfile.createVariable('wspd', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    lvltyp1_var = ncfile.createVariable('lvltyp1', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    lvltyp2_var = ncfile.createVariable('lvltyp2', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    etime_var = ncfile.createVariable('etime', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    pflag_var = ncfile.createVariable('pflag', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    zflag_var = ncfile.createVariable('zflag', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    tflag_var = ncfile.createVariable('tflag', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)
                    dpdp_var = ncfile.createVariable('dpdp', 'f4', ('num_profiles', 'levels'), fill_value=np.nan)

                # Add attributes
                date_var.units = 'YYYYMMDD'
                date_var.long_name = 'Date of sounding'
                
                time_var.units = 'hour'
                time_var.long_name = 'Time of sounding (UTC)'
                
                pressure_var.units = 'hPa'
                pressure_var.long_name = 'Pressure'
                
                gph_var.units = 'm'
                gph_var.long_name = 'Geopotential height'

                rh_var.units = '%'
                rh_var.long_name = 'Relative humidity'

                wdir_var.units = 'degrees from north, 90 = east'
                wdir_var.long_name = 'Wind direction'

                wspd_var.units = 'm/s'
                wspd_var.long_name = 'Wind speed'

                dpdp_var.units = 'degrees C'
                dpdp_var.long_name = 'Dew point depression'
                
                temperature_var.units = 'C'
                temperature_var.long_name = 'Temperature'
                
                # Global attributes
                ncfile.description = f'IGRA sounding data for station {station_id}'
                ncfile.history = f'Created {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                ncfile.source = f'IGRA v2 data for {station_id}'

                # Write data
                if main:
                    date_var[:] = dates
                    time_var[:] = times
                    pressure_var[:, :] = pressure
                    gph_var[:, :] = gph
                    temperature_var[:, :] = temp
                    rh_var[:, :] = rh
                    wdir_var[:, :] = wdir
                    wspd_var[:, :] = wspd
                    dpdp_var[:, :] = dpdp
                else:
                    date_var[:] = dates
                    time_var[:] = times
                    reltime_var[:] = np.array(reltimes)
                    numlev_var[:] = numlevs
                    p_src_var[:] = np.array(p_srcs)
                    np_src_var[:] = np.array(np_srcs)
                    pressure_var[:, :] = pressure
                    gph_var[:, :] = gph
                    temperature_var[:, :] = temp
                    rh_var[:, :] = rh
                    wdir_var[:, :] = wdir
                    wspd_var[:, :] = wspd
                    lvltyp1_var[:, :] = lvltyp1
                    lvltyp2_var[:, :] = lvltyp2
                    etime_var[:, :] = etime
                    pflag_var[:, :] = pflag
                    zflag_var[:, :] = zflag
                    tflag_var[:, :] = tflag
                    dpdp_var[:, :] = dpdp
            
            print(f"Successfully created NetCDF file: {output_file}")
            ds = xr.open_dataset(output_file)
            return ds
        
        elif file_type.lower() in ['pandas', 'df']:
            # Create a list to store all the data
            data = []
            
            # Convert dates and times to datetime objects
            datetimes = []
            for date, time in zip(dates, times):
                # Convert date to string with leading zeros
                date_str = f"{date:08d}"
                
                # If time is invalid (not between 0-23), use default time immediately
                if not (0 <= time <= 23):
                    try:
                        dt = datetime.datetime.strptime(date_str, "%Y%m%d")
                        dt = dt.replace(hour=23, minute=59)
                        datetimes.append(dt)
                    except ValueError:
                        # If we can't parse the date either, use a default datetime
                        dt = datetime.datetime(1900, 1, 1, 23, 59)
                        datetimes.append(dt)
                else:
                    try:
                        # Convert time to string with leading zeros
                        time_str = f"{time:02d}"
                        # Parse the datetime
                        dt = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H")
                        datetimes.append(dt)
                    except ValueError as e:
                        print(f"Warning: Could not parse datetime for date={date}, time={time}: {e}")
                        # Use default datetime if parsing fails
                        dt = datetime.datetime(1900, 1, 1, 23, 59)
                        datetimes.append(dt)
            
            # For each sounding
            for i in range(sounding_count):
                # For each level in the sounding
                for j in range(max_levels):
                    # Skip if all values are NaN
                    if np.isnan(pressure[i, j]) and np.isnan(gph[i, j]) and np.isnan(temp[i, j]):
                        continue
                        
                    row = {
                        'num_profile': i,  # Add the profile number
                        'date': datetimes[i].date(),
                        'time': datetimes[i].time(),
                        'pressure': pressure[i, j],
                        'height': gph[i, j],
                        'temperature': temp[i, j],
                        'relative_humidity': rh[i, j],
                        'wind_direction': wdir[i, j],
                        'wind_speed': wspd[i, j],
                        'dewpoint_depression': dpdp[i, j]
                    }
                    
                    if not main:
                        row.update({
                            'level_type1': lvltyp1[i, j],
                            'level_type2': lvltyp2[i, j],
                            'elapsed_time': etime[i, j],
                            'pressure_flag': pflag[i, j],
                            'height_flag': zflag[i, j],
                            'temperature_flag': tflag[i, j]
                        })
                    
                    data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Filter by date if specified
            if start_date is not None:
                df = df[df['datetime'] >= start_date]
            if end_date is not None:
                df = df[df['datetime'] <= end_date]
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Successfully saved DataFrame to: {output_file}")
            
            return df
        
        else:
            raise ValueError("file_type must be either 'netcdf', 'pandas', or 'df'")
        
    except Exception as e:
        print(f"Error processing data for station {station_id}: {e}")
        return None
    
def open_nc(file_path: Union[str, Path], print_info: bool = True) -> xr.Dataset:
    """
    Open a NetCDF file and optionally print its overview information.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the NetCDF file
    print_info : bool, optional
        Whether to print overview information about the dataset, by default True
        
    Returns
    -------
    xr.Dataset
        The opened NetCDF file as an xarray Dataset
        
    Examples
    --------
    >>> # Open a NetCDF file and print its information
    >>> ds = open_nc("path/to/file.nc")
    
    >>> # Open a NetCDF file without printing information
    >>> ds = open_nc("path/to/file.nc", print_info=False)
    
    >>> # Access data from the dataset
    >>> temperature = ds['temperature']
    >>> print(temperature.values)

    >>> # Access data of the 42nd profile
    >>> temperature = ds['temperature'][42]
    >>> print(temperature.values)
    """
    # Convert string path to Path object if necessary
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {file_path}")
    
    # Open the NetCDF file using xarray
    ds = xr.open_dataset(file_path)
    
    if print_info:
        print("\nNetCDF File Overview:")
        print("=" * 50)
        print(f"File: {file_path}")
        print("\nDimensions:")
        for dim, size in ds.dims.items():
            print(f"  {dim}: {size}")
        
        print("\nVariables:")
        for var in ds.variables:
            var_info = ds[var]
            print(f"  {var}:")
            print(f"    Shape: {var_info.shape}")
            print(f"    Dtype: {var_info.dtype}")
            if 'units' in var_info.attrs:
                print(f"    Units: {var_info.attrs['units']}")
            if 'long_name' in var_info.attrs:
                print(f"    Long name: {var_info.attrs['long_name']}")
        
        if ds.attrs:
            print("\nGlobal Attributes:")
            for attr, value in ds.attrs.items():
                print(f"  {attr}: {value}")
    
    return ds

def open_df(file_path: Union[str, Path], print_info: bool = True) -> pd.DataFrame:
    """
    Open a file as a pandas DataFrame and optionally print its overview information.
    Supports various file formats including CSV, Excel, Parquet, and more.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the data file. Supported formats include:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - Parquet (.parquet)
        - JSON (.json)
        - Pickle (.pkl)
    print_info : bool, optional
        Whether to print overview information about the DataFrame, by default True
        
    Returns
    -------
    pd.DataFrame
        The loaded data as a pandas DataFrame
        
    Examples
    --------
    >>> # Open a CSV file and print its information
    >>> df = open_df("path/to/data.csv")
    
    >>> # Open an Excel file without printing information
    >>> df = open_df("path/to/data.xlsx", print_info=False)
    
    >>> # Access data from the DataFrame
    >>> print(df.head())
    >>> print(df['column_name'].mean())
    """
    # Convert string path to Path object if necessary
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type and load accordingly
    suffix = file_path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(file_path)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif suffix == '.parquet':
        df = pd.read_parquet(file_path)
    elif suffix == '.json':
        df = pd.read_json(file_path)
    elif suffix == '.pkl':
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    if print_info:
        print("\nDataFrame Overview:")
        print("=" * 50)
        print(f"File: {file_path}")
        print(f"\nShape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumns:")
        for col in df.columns:
            col_info = df[col]
            print(f"\n  {col}:")
            print(f"    Type: {col_info.dtype}")
            print(f"    Non-null count: {col_info.count()}")
            print(f"    Null count: {col_info.isna().sum()}")
            if pd.api.types.is_numeric_dtype(col_info):
                print(f"    Mean: {col_info.mean():.2f}")
                print(f"    Std: {col_info.std():.2f}")
                print(f"    Min: {col_info.min():.2f}")
                print(f"    Max: {col_info.max():.2f}")
            elif pd.api.types.is_categorical_dtype(col_info):
                print(f"    Categories: {len(col_info.cat.categories)}")
                print(f"    Most common: {col_info.value_counts().head(3).to_dict()}")
        
        print("\nSample Data:")
        print(df.head())
    
    return df 

def open_data(file_path: Union[str, Path], print_info: bool = True) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Wrapper function to open a file as a pandas DataFrame or xarray Dataset.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the data file.
    print_info : bool, optional
        Whether to print overview information about the DataFrame or Dataset, by default True

    Returns
    -------
    Union[pd.DataFrame, xr.Dataset]
        The loaded data as a pandas DataFrame or xarray Dataset

    Examples
    --------
    >>> # Open a CSV file and print its information
    >>> df = open_data("path/to/data.csv")
    
    >>> # Open a NetCDF file without printing information
    >>> ds = open_data("path/to/data.nc", print_info=False)
    """
    # Convert string path to Path object if necessary
    file_path = Path(file_path)

    if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.parquet', '.json', '.pkl']:
        return open_df(file_path, print_info)
    elif file_path.suffix.lower() in ['.nc']:
        return open_nc(file_path, print_info)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
def filter_by_date_range(df: pd.DataFrame, 
                         start_date: str, 
                         end_date: str, 
                         file_type: Optional[str] = 'df') -> pd.DataFrame:
    """Filter the data by date range.

    Parameters
    ----------
    df : pd.DataFrame
        Sounding data with 'date' column in YYYY-MM-DD format.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    file_type : str, optional
        Type of file to filter ('df' for DataFrame or 'nc' for NetCDF), by default 'df'

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only data between start_date and end_date (inclusive).

    Examples
    --------
    >>> # Filter data for January 2020
    >>> df = read_station_data("USM00072520", file_type='df')
    >>> filtered_df = filter_by_date_range(df, "2020-01-01", "2020-01-31")
    >>> print(filtered_df['date'].min(), filtered_df['date'].max())
    2020-01-01 2020-01-31
    """
    if file_type == 'df':
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
            
        # Convert string dates to datetime objects
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        
        # Ensure date column is in datetime format
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Filter the DataFrame
        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        filtered_df = df[mask].copy()
        
        return filtered_df
    
    elif file_type == 'nc':
        if not isinstance(df, xr.Dataset):
            raise TypeError(f"Expected xarray Dataset, got {type(df).__name__}")
            
        # Convert input dates to YYYYMMDD format
        start_dt = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
        end_dt = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
        
        # Find indices of first and last soundings within date range
        start_idx = np.where(df['date'].values >= start_dt)[0][0]
        end_idx = np.where(df['date'].values <= end_dt)[0][-1]
        
        # Filter the Dataset using indices
        filtered_ds = df.isel(num_profiles=slice(start_idx, end_idx + 1))
        
        return filtered_ds
    
    else:
        raise ValueError("file_type must be either 'df' or 'nc'")

def filter_variables(df: pd.DataFrame, 
                    variables: List[str], 
                    file_type: Optional[str] = 'df') -> Union[pd.DataFrame, xr.Dataset]:
    """Filter the data by variables.

    Parameters
    ----------
    df : Union[pd.DataFrame, xr.Dataset]
        Sounding data with variables to filter.
    variables : List[str]
        List of variable names to keep in the output. For NetCDF files, 'height' will be treated as 'gph'.
    file_type : str, optional
        Type of file to filter ('df' for DataFrame or 'nc' for NetCDF), by default 'df'

    Returns
    -------
    Union[pd.DataFrame, xr.Dataset]
        Filtered data containing only the specified variables plus date, time, and num_profiles.

    Examples
    --------
    >>> # Filter DataFrame to keep only temperature and pressure
    >>> df = read_station_data("USM00072520", file_type='df')
    >>> filtered_df = filter_variables(df, ['temperature', 'pressure'])
    >>> print(filtered_df.columns)
    ['num_profile', 'date', 'time', 'temperature', 'pressure']

    >>> # Filter NetCDF dataset to keep only temperature and height
    >>> ds = read_station_data("USM00072520", file_type='netcdf')
    >>> filtered_ds = filter_variables(ds, ['temperature', 'height'], file_type='nc')
    >>> print(list(filtered_ds.variables))
    ['date', 'time', 'temperature', 'gph']
    """
    if file_type == 'df':
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
            
        # Always keep date, time, and num_profiles columns if they exist
        required_cols = ['num_profile', 'date', 'time']
        valid_vars = [var for var in variables if var in df.columns]
        cols_to_keep = required_cols + valid_vars
        
        # Filter the DataFrame
        filtered_df = df[cols_to_keep].copy()
        
        return filtered_df
    
    elif file_type == 'nc':
        if not isinstance(df, xr.Dataset):
            raise TypeError(f"Expected xarray Dataset, got {type(df).__name__}")
            
        # Always keep date and time variables if they exist
        required_vars = ['date', 'time']
        
        # Handle height/gph synonym
        processed_vars = []
        for var in variables:
            if var == 'height':
                if 'gph' in df.variables:
                    processed_vars.append('gph')
            else:
                processed_vars.append(var)
                
        valid_vars = [var for var in processed_vars if var in df.variables]
        vars_to_keep = required_vars + valid_vars
        
        # Filter the Dataset
        filtered_ds = df[vars_to_keep]
        
        return filtered_ds
    
    else:
        raise ValueError("file_type must be either 'df' or 'nc'")
    
def read_station_locations(save_file: bool = True, start_year: int = 1900, end_year: int = 2030) -> pd.DataFrame:
    """Download and parse the IGRA station list.

    Downloads the station list from NOAA's IGRA archive and parses it into a DataFrame
    containing station metadata including location, elevation, and data availability.

    Parameters
    ----------
    save_file : bool, optional
        Whether to save the DataFrame to a CSV file, by default True.
        If True, saves to 'igra_stations.csv' in the current directory.
    start_year : int, optional
        First year of data availability to include in the DataFrame, by default 1900.
    end_year : int, optional
        Last year of data availability to include in the DataFrame, by default 2025.

    Returns
    -------
    pd.DataFrame
        DataFrame containing station information with columns:
        - station_id: Station identifier
        - latitude: Station latitude in degrees
        - longitude: Station longitude in degrees
        - elevation: Station elevation in meters
        - name: Station name
        - first_year: First year of data availability
        - last_year: Last year of data availability

    Examples
    --------
    >>> # Get station locations and save to file
    >>> stations_df = read_station_locations()
    >>> print(stations_df.head())
    
    >>> # Get station locations without saving
    >>> stations_df = read_station_locations(save_file=False)
    >>> print(stations_df.head())
    
    >>> # Plot station locations
    >>> fig = px.scatter_geo(stations_df,
    ...                      lat='latitude',
    ...                      lon='longitude',
    ...                      hover_data=['station_id', 'name'],
    ...                      projection='natural earth',
    ...                      title='IGRA Station Locations')
    >>> fig.show()
    """
    
    try:
        # Download the station list
        print("Downloading IGRA station list...")
        response = requests.get(metadata.IGRA_STATION_LIST_URL)
        response.raise_for_status()
        
        # Parse the text content
        lines = response.text.splitlines()
        
        # Skip header lines
        data_lines = [line for line in lines if line.strip() and not line.startswith('-')]
        
        # Parse each line according to the fixed-width format
        stations = []
        for line in data_lines:
            try:
                if int(line[72:76].strip()) >= start_year and int(line[77:81].strip()) <= end_year:
                    station = {
                        'station_id': line[0:11].strip(),
                        'latitude': float(line[12:20].strip()),
                        'longitude': float(line[21:30].strip()),
                        'elevation': float(line[31:37].strip()),
                        'state': line[38:40].strip(),
                        'name': line[41:71].strip(),
                        'first_year': int(line[72:76].strip()),
                        'last_year': int(line[77:81].strip()),
                        'nobs': int(line[82:88].strip())
                    }
                    stations.append(station)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line}")
                print(f"Error: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(stations)
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['latitude', 'longitude', 'elevation', 'first_year', 'last_year', 'nobs']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        print(f"Successfully parsed {len(df)} stations")
        
        # Save to CSV if requested
        if save_file:
            output_file = 'igra_stations.csv'
            df.to_csv(output_file, index=False)
            print(f"Saved station list to {output_file}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading station list: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error processing station list: {e}")
        return pd.DataFrame()

def plot_station_map(color_by: str = 'none', 
                     start_year: int = 1900, 
                     end_year: int = 2025, 
                     lat_range: Tuple[float, float] = (-90, 90),
                     lon_range: Tuple[float, float] = (-180, 180)):
    """
    Displays an interactive map of IGRA stations using Plotly.
    
    Parameters
    ----------
    color_by : str, optional
        Variable to use for coloring the stations. Options are:
        - 'none': No coloring (default)
        - 'elevation': Color stations by their elevation
        - 'last_year': Color stations by their last year of data
        - 'first_year': Color stations by their first year of data
        - 'nobs': Color stations by number of observations
    start_year : int, optional
        First year of data availability to include in the map, by default 1900
    end_year : int, optional
        Last year of data availability to include in the map, by default 2025
    lat_range : Tuple[float, float], optional
        Range of latitudes to display (min_lat, max_lat), by default (-90, 90)
    lon_range : Tuple[float, float], optional
        Range of longitudes to display (min_lon, max_lon), by default (-180, 180)

    Examples
    --------
    >>> # Display the map colored by elevation (default)
    >>> plot_station_map()
    
    >>> # Display the map colored by last year of data
    >>> plot_station_map(color_by='last_year')
    
    >>> # Display the map colored by number of observations
    >>> plot_station_map(color_by='nobs')
    
    >>> # Display stations in a specific region
    >>> plot_station_map(lat_range=(30, 50), lon_range=(-130, -70))  # North America
    """

    stations_df = read_station_locations(save_file=False)
    
    # Filter stations by year range
    stations_df = stations_df[
        (stations_df['first_year'] >= start_year) & 
        (stations_df['last_year'] <= end_year)
    ]
    
    # Filter stations by latitude and longitude range
    stations_df = stations_df[
        (stations_df['latitude'] >= lat_range[0]) &
        (stations_df['latitude'] <= lat_range[1]) &
        (stations_df['longitude'] >= lon_range[0]) &
        (stations_df['longitude'] <= lon_range[1])
    ]

    if color_by == 'none':
        fig = px.scatter_geo(
            stations_df,
            lat='latitude',
            lon='longitude',
            hover_data=['station_id', 'name', 'elevation', 'first_year', 'last_year', 'nobs'],
            projection='natural earth',
            title='IGRA Station Locations'
        )
        fig.update_traces(marker=dict(size=6, color='blue'))
    else:
        # Set up color mapping based on color_by parameter
        color_mapping = {
            'elevation': {
                'data': stations_df['elevation'],
                'colorscale': 'viridis',
                'title': 'Elevation (m)'
            },
            'first_year': {
                'data': stations_df['first_year'],
                'colorscale': 'bluered',
                'title': 'First Year of Data'
            },
            'last_year': {
                'data': stations_df['last_year'],
                'colorscale': 'bluered',
                'title': 'Last Year of Data'
            },
            'nobs': {
                'data': stations_df['nobs'],
                'colorscale': 'viridis',
                'title': 'Number of Observations'
            }
        }

        if color_by not in color_mapping:
            raise ValueError(f"color_by must be one of {list(color_mapping.keys())}")

        # Get the color mapping for the selected variable
        color_info = color_mapping[color_by]

        # Create the figure
        fig = px.scatter_geo(
            stations_df,
            lat='latitude',
            lon='longitude',
            color=color_info['data'],
            color_continuous_scale=color_info['colorscale'],
            hover_data=['station_id', 'name', 'elevation', 'first_year', 'last_year', 'nobs'],
            projection='natural earth',
            title='IGRA Station Locations'
        )
        
        # Update marker appearance
        fig.update_traces(
            marker=dict(
                size=6,
                opacity=0.6,
                line=dict(width=0)
            )
        )

    # Update layout to show the full global map
    fig.update_layout(
        geo=dict(
            showland=True,
            showcoastlines=True,
            showcountries=True,
            showocean=True,
            oceancolor='rgb(204, 229, 255)',
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(128, 128, 128)',
            countrycolor='rgb(128, 128, 128)',
            projection=dict(
                type='natural earth'
            )
        )
    )
    
    fig.show()

def filter_stations(start_year: int = 1900, 
                     end_year: int = 2025, 
                     lat_range: Tuple[float, float] = (-1000, 1000),
                     lon_range: Tuple[float, float] = (-1000, 1000)) -> List[str]:
    """Filter station data by year, latitude, and longitude range.
    
    Parameters
    ----------
    start_year : int, optional
        First year of data availability to include, by default 1900
    end_year : int, optional
        Last year of data availability to include, by default 2025
    lat_range : Tuple[float, float], optional
        Range of latitudes to include (min_lat, max_lat), by default (-90, 90)
    lon_range : Tuple[float, float], optional
        Range of longitudes to include (min_lon, max_lon), by default (-180, 180)
        
    Returns
    -------
    List[str]
        List of station IDs that meet the filtering criteria
        
    Examples
    --------
    >>> # Get all stations in Antarctica
    >>> antarctic_stations = filter_stations(lat_range=(-90, -60))
    >>> print(f"Found {len(antarctic_stations)} stations in Antarctica")
    
    >>> # Get stations in North America from 2000 onwards
    >>> na_stations = filter_stations(
    ...     start_year=2000,
    ...     lat_range=(30, 50),
    ...     lon_range=(-130, -70)
    ... )
    >>> print(f"Found {len(na_stations)} stations in North America since 2000")
    
    >>> # Get stations in a specific region
    >>> ny_stations = filter_stations(
    ...     lat_range=(40, 45),
    ...     lon_range=(-75, -70)
    ... )
    >>> print(f"Found {len(ny_stations)} stations in the New York area")
    """
    stations_df = read_station_locations(save_file=False)
    
    # Filter stations by year range
    stations_df = stations_df[
        (stations_df['first_year'] >= start_year) & 
        (stations_df['last_year'] <= end_year)
    ]
    
    # Filter stations by latitude and longitude range
    stations_df = stations_df[
        (stations_df['latitude'] >= lat_range[0]) &
        (stations_df['latitude'] <= lat_range[1]) &
        (stations_df['longitude'] >= lon_range[0]) &
        (stations_df['longitude'] <= lon_range[1])
    ]

    return stations_df['station_id'].tolist()

def _filter_invalid_values(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filter out invalid values from paired arrays.
    
    Parameters
    ----------
    x : np.ndarray
        First array of values
    y : np.ndarray
        Second array of values
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Filtered x and y arrays with invalid values removed
    """
    # Define invalid values
    invalid_values = [np.nan, -9999, -8888, 9999, 8888]
    
    # Create mask for valid values
    mask = ~np.isnan(x) & ~np.isnan(y)
    for val in invalid_values:
        mask &= (x != val) & (y != val)
    
    return x[mask], y[mask]

def interp_data(data: Union[pd.DataFrame, xr.Dataset, nc.Dataset], 
                             index_variable: str, 
                             variable: str, 
                             min_index: float,
                             max_index: float,
                             step_size: float,
                             start_date: Optional[datetime.datetime] = None,
                             end_date: Optional[datetime.datetime] = None,
                             fill_value: Optional[float] = None
                             ) -> Union[pd.DataFrame, xr.Dataset]:
    """Interpolate station data onto a uniform grid.

    This function creates a uniform grid of the index variable and interpolates
    the specified variable onto that grid. It supports both DataFrame and NetCDF data formats.

    One-dimensional linear interpolation is used.

    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset, nc.Dataset]
        Input data to interpolate. Can be:
        - pandas DataFrame
        - xarray Dataset
        - netCDF4 Dataset
    index_variable : str
        Name of the variable to use as the index for interpolation (e.g., 'height' or 'pressure')
    variable : str
        Name of the variable to interpolate (e.g., 'temperature' or 'wind_speed')
    min_index : float
        Minimum value for the uniform grid
    max_index : float
        Maximum value for the uniform grid
    step_size : float
        Step size between grid points
    start_date : Optional[datetime.datetime], optional
        Start date for filtering data, by default None
    end_date : Optional[datetime.datetime], optional
        End date for filtering data, by default None
    fill_value : Optional[float], optional
        Value to use for extrapolation, by default None

    Returns
    -------
    Union[pd.DataFrame, xr.Dataset]
        Interpolated data with the same format as the input, but with uniform grid spacing

    Examples
    --------
    >>> # Interpolate temperature data onto a uniform height grid
    >>> df = read_station_data("USM00072520", file_type='df')
    >>> interpolated_df = interpolate_station_data(
    ...     df,
    ...     index_variable='height',
    ...     variable='temperature',
    ...     min_index=0,
    ...     max_index=10000,
    ...     step_size=100,
    ...     method='linear'
    ... )
    >>> print(interpolated_df['height'].unique())  # Will show uniform height levels
    [0, 100, 200, ..., 10000]

    >>> # Interpolate wind speed onto a uniform pressure grid
    >>> ds = read_station_data("USM00072520", file_type='netcdf')
    >>> interpolated_ds = interpolate_station_data(
    ...     ds,
    ...     index_variable='pressure',
    ...     variable='wind_speed',
    ...     min_index=1000,
    ...     max_index=100,
    ...     step_size=10,
    ...     method='cubic'
    ... )
    >>> print(interpolated_ds['pressure'].values)  # Will show uniform pressure levels
    [1000, 990, 980, ..., 100]
    """
    # Filter by date range if specified
    if start_date is not None or end_date is not None:
        if isinstance(data, pd.DataFrame):
            data = filter_by_date_range(data, start_date, end_date, file_type='df')
        else:
            data = filter_by_date_range(data, start_date, end_date, file_type='nc')
    
    # Create uniform grid
    grid = np.arange(min_index, max_index + step_size, step_size)
            
    if isinstance(data, pd.DataFrame):
        if variable not in data.columns:
            raise ValueError(f"Variable '{variable}' not found in DataFrame")
            
        if index_variable not in data.columns:
            raise ValueError(f"Index variable '{index_variable}' not found in DataFrame")
            
        # Create a copy to avoid modifying the original
        interpolated_data = data.copy()
        
        # Group by profile number to interpolate within each profile
        grouped = interpolated_data.groupby('num_profile')
        
        # Create empty list to store interpolated profiles
        interpolated_profiles = []
        
        # Interpolate each profile onto the uniform grid
        for profile_num, profile in grouped:
            # Get the original values
            x = profile[index_variable].values
            y = profile[variable].values
            
            # Filter out invalid values
            x, y = _filter_invalid_values(x, y)
            
            if len(x) > 1:  # Need at least 2 points for interpolation
                # Interpolate onto the uniform grid
                interpolated_values = np.interp(grid, x, y, left=fill_value, right=fill_value)
                
                # Create a new DataFrame for this profile
                profile_df = pd.DataFrame({
                    'num_profile': profile_num,
                    'date': profile['date'].iloc[0],
                    'time': profile['time'].iloc[0],
                    index_variable: grid,
                    variable: interpolated_values
                })
                
                interpolated_profiles.append(profile_df)
        
        # Combine all interpolated profiles
        if interpolated_profiles:
            return pd.concat(interpolated_profiles, ignore_index=True)
        else:
            return pd.DataFrame()
        
    elif isinstance(data, (xr.Dataset, nc.Dataset)):
        # Convert netCDF4 Dataset to xarray Dataset if needed
        if isinstance(data, nc.Dataset):
            data = xr.Dataset.from_dict(data.variables)
            
        # Handle height/gph synonym
        var_name = 'gph' if variable == 'height' else variable
        index_name = 'gph' if index_variable == 'height' else index_variable
        
        if var_name not in data.variables:
            raise ValueError(f"Variable '{variable}' not found in Dataset")
            
        if index_name not in data.variables:
            raise ValueError(f"Index variable '{index_variable}' not found in Dataset")
            
        # Create a copy to avoid modifying the original
        interpolated_data = data.copy()
        
        # Create new coordinate for the uniform grid
        new_coord = xr.DataArray(
            grid,
            dims=['levels'],
            coords={'levels': grid}
        )
        
        # Create a new dataset with the interpolated values
        interpolated_values = []
        for profile_idx in range(len(interpolated_data.num_profiles)):
            # Get the original values for this profile
            x = interpolated_data[index_name].isel(num_profiles=profile_idx).values
            y = interpolated_data[var_name].isel(num_profiles=profile_idx).values
            
            # Filter out invalid values
            x, y = _filter_invalid_values(x, y)
            
            if len(x) > 1:  # Need at least 2 points for interpolation
                # Interpolate onto the uniform grid
                interpolated_values.append(np.interp(grid, x, y, left=fill_value, right=fill_value))
            else:
                interpolated_values.append(np.full_like(grid, np.nan))
        
        # Create new dataset with interpolated values
        new_data = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.array(interpolated_values),
                    dims=['num_profiles', 'levels'],
                    coords={
                        'num_profiles': interpolated_data.num_profiles,
                        'levels': grid
                    }
                ),
                index_name: xr.DataArray(
                    np.tile(grid, (len(interpolated_data.num_profiles), 1)),
                    dims=['num_profiles', 'levels'],
                    coords={
                        'num_profiles': interpolated_data.num_profiles,
                        'levels': grid
                    }
                )
            }
        )
        
        # Copy over date and time variables
        new_data['date'] = interpolated_data['date']
        new_data['time'] = interpolated_data['time']
        
        return new_data
    
    else:
        raise TypeError(f"Expected pandas DataFrame, xarray Dataset, or netCDF4 Dataset, got {type(data).__name__}")

def interp_data_to_pressure_levels(data: Union[pd.DataFrame, xr.Dataset, nc.Dataset], 
                               variable: str,
                               start_date: Optional[datetime.datetime] = None,
                               end_date: Optional[datetime.datetime] = None,
                               fill_value: Optional[float] = None
                               ) -> Union[pd.DataFrame, xr.Dataset]:
    """Interpolate station data onto standard pressure levels.

    This function interpolates the specified variable onto standard pressure levels
    (1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10 hPa).
    It supports both DataFrame and NetCDF data formats.

    For pressure interpolation, log-pressure interpolation is used to better handle
    the exponential decrease of pressure with height.

    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset, nc.Dataset]
        Input data to interpolate. Can be:
        - pandas DataFrame
        - xarray Dataset
        - netCDF4 Dataset
    variable : str
        Name of the variable to interpolate (e.g., 'temperature' or 'wind_speed')
    start_date : Optional[datetime.datetime], optional
        Start date for filtering data, by default None
    end_date : Optional[datetime.datetime], optional
        End date for filtering data, by default None
    fill_value : Optional[float], optional
        Value to use for extrapolation, by default None

    Returns
    -------
    Union[pd.DataFrame, xr.Dataset]
        Interpolated data with the same format as the input, but with standard pressure levels

    Examples
    --------
    >>> # Interpolate temperature to standard pressure levels
    >>> df = read_station_data("USM00072520", file_type='df')
    >>> interpolated_df = interp_data_to_pressure_levels(
    ...     df,
    ...     variable='temperature'
    ... )
    >>> print(interpolated_df['pressure'].unique())  # Will show standard pressure levels
    [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]

    >>> # Interpolate wind speed to standard pressure levels
    >>> ds = read_station_data("USM00072520", file_type='netcdf')
    >>> interpolated_ds = interp_data_to_pressure_levels(
    ...     ds,
    ...     variable='wind_speed'
    ... )
    >>> print(interpolated_ds['pressure'].values)  # Will show standard pressure levels
    [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
    """
    # Filter by date range if specified
    if start_date is not None or end_date is not None:
        if isinstance(data, pd.DataFrame):
            data = filter_by_date_range(data, start_date, end_date, file_type='df')
        else:
            data = filter_by_date_range(data, start_date, end_date, file_type='nc')
    
    # Create uniform grid
    grid = [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
    log_grid = np.log(grid)
            
    if isinstance(data, pd.DataFrame):
        if variable not in data.columns:
            raise ValueError(f"Variable '{variable}' not found in DataFrame")
            
        # Create a copy to avoid modifying the original
        interpolated_data = data.copy()
        
        # Group by profile number to interpolate within each profile
        grouped = interpolated_data.groupby('num_profile')
        
        # Create empty list to store interpolated profiles
        interpolated_profiles = []
        
        # Interpolate each profile onto the uniform grid
        for profile_num, profile in grouped:
            # Get the original values
            x = profile['pressure'].values
            y = profile[variable].values
            
            # Filter out invalid values
            x, y = _filter_invalid_values(x, y)
            log_x = np.log(x)
            sort_idx = np.argsort(log_x)
            log_x = log_x[sort_idx]
            y = y[sort_idx]

            
            if len(x) > 1:  # Need at least 2 points for interpolation
                interpolated_values = np.interp(log_grid, log_x, y, left=fill_value, right=fill_value)
                
                # Create a new DataFrame for this profile
                profile_df = pd.DataFrame({
                    'num_profile': profile_num,
                    'date': profile['date'].iloc[0],
                    'time': profile['time'].iloc[0],
                    'pressure': grid,
                    variable: interpolated_values
                })
                
                interpolated_profiles.append(profile_df)
        
        # Combine all interpolated profiles
        if interpolated_profiles:
            return pd.concat(interpolated_profiles, ignore_index=True)
        else:
            return pd.DataFrame()
        
    elif isinstance(data, (xr.Dataset, nc.Dataset)):
        # Convert netCDF4 Dataset to xarray Dataset if needed
        if isinstance(data, nc.Dataset):
            data = xr.Dataset.from_dict(data.variables)
            
        # Handle height/gph synonym
        var_name = 'gph' if variable == 'height' else variable
        
        if var_name not in data.variables:
            raise ValueError(f"Variable '{variable}' not found in Dataset")
            
        # Create a copy to avoid modifying the original
        interpolated_data = data.copy()
        
        # Create new coordinate for the uniform grid
        new_coord = xr.DataArray(
            grid,
            dims=['levels'],
            coords={'levels': grid}
        )
        
        # Create a new dataset with the interpolated values
        interpolated_values = []
        for profile_idx in range(len(interpolated_data.num_profiles)):
            # Get the original values for this profile
            x = interpolated_data['pressure'].isel(num_profiles=profile_idx).values
            y = interpolated_data[var_name].isel(num_profiles=profile_idx).values
            
            # Filter out invalid values
            x, y = _filter_invalid_values(x, y)
            log_x = np.log(x)
            sort_idx = np.argsort(log_x)
            log_x = log_x[sort_idx]
            y = y[sort_idx]
            
            if len(x) > 1:  # Need at least 2 points for interpolation
                # Interpolate onto the uniform grid
                interpolated_values.append(np.interp(log_grid, log_x, y, left=fill_value, right=fill_value))
            else:
                interpolated_values.append(np.full_like(log_grid, np.nan))
        
        # Create new dataset with interpolated values
        new_data = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.array(interpolated_values),
                    dims=['num_profiles', 'levels'],
                    coords={
                        'num_profiles': interpolated_data.num_profiles,
                        'levels': grid
                    }
                ),
                'pressure': xr.DataArray(
                    np.tile(grid, (len(interpolated_data.num_profiles), 1)),
                    dims=['num_profiles', 'levels'],
                    coords={
                        'num_profiles': interpolated_data.num_profiles,
                        'levels': grid
                    }
                )
            }
        )
        
        # Copy over date and time variables
        new_data['date'] = interpolated_data['date']
        new_data['time'] = interpolated_data['time']
        
        return new_data
    
    else:
        raise TypeError(f"Expected pandas DataFrame, xarray Dataset, or netCDF4 Dataset, got {type(data).__name__}")

def get_availability(data: Union[pd.DataFrame, xr.Dataset]) -> Optional[Dict]:
    """
    Get the availability of IGRA data from a DataFrame or Dataset.

    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        Input data containing the profiles

    Returns
    -------
    Optional[Dict]
        Nested dictionary containing availability information organized as:
        {
            year: {
                month: {
                    day: ['HH:MM:SS', ...]
                }
            }
        }

    Examples
    --------
    >>> # Get availability from a DataFrame
    >>> df = open_data("USM00072520-main.csv")
    >>> availability = get_availability(df)
    >>> # Access times for a specific date
    >>> times = availability[2020][1][1]  # Times for January 1, 2020
    >>> print(f"Available times: {times}")
    """
    try:
        availability = []
        
        if isinstance(data, pd.DataFrame):
            # Extract date and time information from DataFrame
            for _, row in data.groupby('num_profile').first().iterrows():
                # Parse date string (YYYY-MM-DD)
                date_parts = str(row['date']).split('-')
                year = int(date_parts[0])
                month = int(date_parts[1])
                day = int(date_parts[2])
                
                # Get time string directly (HH:MM:SS)
                time_str = str(row['time'])
                availability.append([year, month, day, time_str])
                
        elif isinstance(data, (xr.Dataset, nc.Dataset)):
            # Extract date and time information from Dataset
            dates = data['date'].values
            times = data['time'].values
            
            for date, time in zip(dates, times):
                # Convert YYYYMMDD to year, month, day
                year = int(date // 10000)
                month = int((date % 10000) // 100)
                day = int(date % 100)
                # Format time as HH:MM:SS
                time_str = f"{int(time):02d}:00:00"
                availability.append([year, month, day, time_str])
                
        else:
            raise TypeError(f"Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")

        # Create nested dictionary structure
        nested_availability = {}
        
        # Sort availability by year, month, day, time
        availability.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        
        for year, month, day, time_str in availability:
            # Initialize year if not exists
            if year not in nested_availability:
                nested_availability[year] = {}
            
            # Initialize month if not exists
            if month not in nested_availability[year]:
                nested_availability[year][month] = {}
            
            # Initialize day if not exists
            if day not in nested_availability[year][month]:
                nested_availability[year][month][day] = []
            
            # Add time if not already in the list
            if time_str not in nested_availability[year][month][day]:
                nested_availability[year][month][day].append(time_str)
                # Sort times
                nested_availability[year][month][day].sort()

        return nested_availability
    
    except Exception as e:
        print(f"Error processing availability data: {e}")
        return None

def plot_profile(data: Union[pd.DataFrame, xr.Dataset],
                x_variable: str,
                y_variable: str,
                date: str,  # Format: YYYY-MM-DD
                time: str,  # Format: HH:MM:SS
                figsize: Tuple[int, int] = (10, 8),
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                grid: bool = True,
                show: bool = True) -> Optional[plt.Figure]:
    """Plot a vertical profile for a specific date and time.

    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        Input data containing the profile
    x_variable : str
        Name of the variable to plot on x-axis
    y_variable : str
        Name of the variable to plot on y-axis (typically height or pressure)
    date : str
        Date of the profile to plot in YYYY-MM-DD format
    time : str
        Time of the profile to plot in HH:MM:SS format
    figsize : Tuple[int, int], optional
        Figure size in inches (width, height), by default (10, 8)
    title : Optional[str], optional
        Plot title, by default None
    xlabel : Optional[str], optional
        X-axis label, by default None
    ylabel : Optional[str], optional
        Y-axis label, by default None
    grid : bool, optional
        Whether to show grid lines, by default True
    show : bool, optional
        Whether to display the plot, by default True

    Returns
    -------
    Optional[plt.Figure]
        The matplotlib figure object if show=False, None otherwise

    Examples
    --------
    >>> # Plot temperature profile for a specific date and time
    >>> fig = plot_profile(
    ...     data,
    ...     x_variable='temperature',
    ...     y_variable='height',
    ...     date='2020-01-01',
    ...     time='12:00:00',
    ...     title='Temperature Profile'
    ... )
    """
    # Define units for common variables
    units = {
        'temperature': '°C',
        'height': 'm',
        'pressure': 'hPa',
        'relative_humidity': '%',
        'wind_speed': 'm/s',
        'wind_direction': '°',
        'dewpoint_depression': '°C',
        'gph': 'm'
    }

    # Get availability data
    availability = get_availability(data)
    if availability is None:
        print("Error: Could not determine data availability")
        return None

    # Parse input date and time
    try:
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:10])
    except (ValueError, IndexError) as e:
        print(f"Error parsing date/time: {e}")
        print("Date should be in YYYY-MM-DD format")
        print("Time should be in HH:MM:SS format")
        return None

    # Check if the requested date/time exists in the data
    if year not in availability:
        print(f"No data available for year {year}")
        print(f"Available years: {sorted(availability.keys())}")
        return None

    if month not in availability[year]:
        print(f"No data available for {year}-{month:02d}")
        print(f"Available months: {sorted(availability[year].keys())}")
        return None

    if day not in availability[year][month]:
        print(f"No data available for {year}-{month:02d}-{day:02d}")
        print(f"Available days: {sorted(availability[year][month].keys())}")
        return None

    if time not in availability[year][month][day]:
        print(f"No data available for {year}-{month:02d}-{day:02d} {time}")
        print(f"Available times: {sorted(availability[year][month][day])}")
        return None

    if isinstance(data, pd.DataFrame):
        # Filter data for the specific date and time
        profile_data = data[
            (data['date'] == date) &
            (data['time'] == time)
        ]

        if len(profile_data) == 0:
            print(f"No profile found for {date} {time}")
            return None

        # Filter out invalid values (NaN, -9999, -8888)
        valid_mask = (
            ~profile_data[x_variable].isna() & 
            ~profile_data[y_variable].isna() &
            (profile_data[x_variable] != -9999) &
            (profile_data[x_variable] != -8888) &
            (profile_data[y_variable] != -9999) &
            (profile_data[y_variable] != -8888)
        )
        x_data = profile_data[x_variable][valid_mask]
        y_data = profile_data[y_variable][valid_mask]

        if len(x_data) == 0 or len(y_data) == 0:
            print(f"Error: No valid data points found for {x_variable} vs {y_variable}")
            return None

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the profile
        ax.plot(x_data, y_data)
        
        # Set labels and title with units
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            unit = units.get(x_variable, '')
            ax.set_xlabel(f"{x_variable.capitalize()} [{unit}]")
            
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            unit = units.get(y_variable, '')
            ax.set_ylabel(f"{y_variable.capitalize()} [{unit}]")
            
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{x_variable.capitalize()} vs {y_variable.capitalize()} Profile\n{date} {time}")
            
        if grid:
            ax.grid(True)
            
        # Invert y-axis if y_variable is pressure
        if y_variable.lower() in ['pressure', 'p']:
            ax.invert_yaxis()
            
        if show:
            plt.show()
            return None
        else:
            return fig

    elif isinstance(data, (xr.Dataset, nc.Dataset)):
        # Convert date to YYYYMMDD format for xarray
        date_int = int(date.replace('-', ''))
        time_int = int(time[:2])

        # Filter data for the specific date and time
        profile_data = data.where(
            (data['date'] == date_int) &
            (data['time'] == time_int),
            drop=True
        )

        if len(profile_data['num_profiles']) == 0:
            print(f"No profile found for {date} {time}")
            return None

        # Handle height/gph synonym
        x_name = 'gph' if x_variable == 'height' else x_variable
        y_name = 'gph' if y_variable == 'height' else y_variable

        # Filter out invalid values (NaN, -9999, -8888)
        valid_mask = (
            ~np.isnan(profile_data[x_name]) & 
            ~np.isnan(profile_data[y_name]) &
            (profile_data[x_name] != -9999) &
            (profile_data[x_name] != -8888) &
            (profile_data[y_name] != -9999) &
            (profile_data[y_name] != -8888)
        )
        x_data = profile_data[x_name].where(valid_mask, drop=True)
        y_data = profile_data[y_name].where(valid_mask, drop=True)

        if len(x_data) == 0 or len(y_data) == 0:
            print(f"Error: No valid data points found for {x_variable} vs {y_variable}")
            return None

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the profile - convert xarray DataArrays to numpy arrays
        ax.plot(x_data.values.flatten(), y_data.values.flatten())
        
        # Set labels and title with units
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            unit = units.get(x_variable, '')
            ax.set_xlabel(f"{x_variable.capitalize()} [{unit}]")
            
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            unit = units.get(y_variable, '')
            ax.set_ylabel(f"{y_variable.capitalize()} [{unit}]")
            
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{x_variable.capitalize()} vs {y_variable.capitalize()} Profile\n{date} {time}")
            
        if grid:
            ax.grid(True)
            
        # Invert y-axis if y_variable is pressure
        if y_variable.lower() in ['pressure', 'p']:
            ax.invert_yaxis()
            
        if show:
            plt.show()
            return None
        else:
            return fig

    else:
        raise TypeError(f"Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")
    
def get_profile(data: Union[pd.DataFrame, xr.Dataset],
                date: str,
                time: str) -> Optional[Union[pd.DataFrame, xr.Dataset]]:
    """
    Get a profile for a specific date and time.

    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        Input data containing the profiles
    date : str
        Date in YYYY-MM-DD format
    time : str
        Time in HH:MM:SS format

    Returns
    -------
    Optional[Union[pd.DataFrame, xr.Dataset]]
        Data containing only the profile for the specified date and time.
        Returns None if no profile is found.

    Examples
    --------
    >>> # Get a profile from a DataFrame
    >>> df = open_data("USM00072520-main.csv")
    >>> profile = get_profile(df, '2020-01-01', '12:00:00')
    >>> print(profile.head())

    >>> # Get a profile from a Dataset
    >>> ds = open_data("USM00072520-main.nc")
    >>> profile = get_profile(ds, '2020-01-01', '12:00:00')
    >>> print(profile)
    """
    try:
        # Get availability data
        availability = get_availability(data)
        if availability is None:
            print("Error: Could not determine data availability")
            return None

        # Parse input date and time
        try:
            year = int(date[:4])
            month = int(date[5:7])
            day = int(date[8:10])
        except (ValueError, IndexError) as e:
            print(f"Error parsing date/time: {e}")
            print("Date should be in YYYY-MM-DD format")
            print("Time should be in HH:MM:SS format")
            return None

        # Check if the requested date/time exists in the data
        if year not in availability:
            print(f"No data available for year {year}")
            print(f"Available years: {sorted(availability.keys())}")
            return None

        if month not in availability[year]:
            print(f"No data available for {year}-{int(month):02d}")
            print(f"Available months: {sorted(availability[year].keys())}")
            return None

        if day not in availability[year][month]:
            print(f"No data available for {year}-{int(month):02d}-{int(day):02d}")
            print(f"Available days: {sorted(availability[year][month].keys())}")
            return None

        if time not in availability[year][month][day]:
            print(f"No data available for {year}-{int(month):02d}-{int(day):02d} {time}")
            print(f"Available times: {sorted(availability[year][month][day])}")
            return None

        # If we get here, we know the date/time exists in the data
        if isinstance(data, pd.DataFrame):
            # Filter data for the specific date and time
            profile_data = data[
                (data['date'] == date) &
                (data['time'] == time)
            ]

            if len(profile_data) == 0:
                print(f"No profile found for {date} {time}")
                return None

            return profile_data

        elif isinstance(data, (xr.Dataset, nc.Dataset)):
            # Convert date to YYYYMMDD format for xarray
            date_int = int(date.replace('-', ''))
            time_int = int(time[:2])

            # Filter data for the specific date and time
            profile_data = data.where(
                (data['date'] == date_int) &
                (data['time'] == time_int),
                drop=True
            )

            if len(profile_data['num_profiles']) == 0:
                print(f"No profile found for {date} {time}")
                return None

            return profile_data

        else:
            raise TypeError(f"Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")

    except Exception as e:
        print(f"Error getting profile: {e}")
        return None


