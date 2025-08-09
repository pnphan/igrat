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
from scipy import interpolate
import datetime
import tempfile

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
            
        os.makedirs(output_dir, exist_ok=True)
        
        base_url = metadata.IGRA_FILES_URL
        zip_filename = f"{station_id}-data.txt.zip"
        zip_url = urljoin(base_url, zip_filename)
        
        print(f"Downloading data for station {station_id}...")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        zip_buffer = io.BytesIO()
        
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {station_id}") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    zip_buffer.write(chunk)
                    pbar.update(len(chunk))
        
        zip_buffer.seek(0)
        
        print(f"Extracting data for station {station_id}...")
        with zipfile.ZipFile(zip_buffer) as zip_ref:
            txt_file_name = None
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.txt'):
                    txt_file_name = file_info.filename
                    break
            
            if txt_file_name:
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
                      download: bool = True,
                      download_dir: Optional[str] = None,
                     file_type: Optional[str] = 'netcdf',
                     file_name: Optional[str] = None) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Read and parse a single station's IGRA data file.
    
    Args:
        station_id: Station ID (e.g., "USM00072520" for Albany, NY)
        path: Optional path to the station's data file
        main: Whether to include only main variables (True) or all variables (False)
        download: Whether to save the data to a file or just return the data
        download_dir: Directory to save the output file 
        file_type: Optional file type to return, either 'netcdf', 'pandas', or 'df'
        file_name: Optional name of the output file
    Returns:
        If file_type is 'netcdf':
            xarray.Dataset containing sounding data with dimensions:
            - num_profiles: Number of soundings
            - levels: Number of levels in each sounding
            - variables: Pressure, height, temperature, dewpoint, wind_direction, wind_speed, relative_humidity
            Data is saved as a netcdf file if download=True. 
            
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
            Data is saved as a csv file if download=True.

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
        >>> df = read_station_data(station, 
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

        if file_name is None:
            file_name = station_id

        if file_type.lower() in ['netcdf', 'nc']:
            if main:
                output_file = os.path.join(download_dir, f"{file_name}-main.nc")
            else:
                output_file = os.path.join(download_dir, f"{file_name}-full.nc")
        else:  # pandas/df
            if main:
                output_file = os.path.join(download_dir, f"{file_name}-main.csv")
            else:
                output_file = os.path.join(download_dir, f"{file_name}-full.csv")

        if path is None:
            base_url = metadata.IGRA_FILES_URL

            zip_filename = f"{station_id}-data.txt.zip"
            zip_url = urljoin(base_url, zip_filename)
            
            try:
                print(f"Downloading data for station {station_id}...")
                response = requests.get(zip_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                zip_buffer = io.BytesIO()
                
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {station_id}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            zip_buffer.write(chunk)
                            pbar.update(len(chunk))
                
                zip_buffer.seek(0)
                
                print(f"Extracting data for station {station_id}...")
                with zipfile.ZipFile(zip_buffer) as zip_ref:
                    txt_file_name = None
                    for file_info in zip_ref.infolist():
                        if file_info.filename.endswith('.txt'):
                            txt_file_name = file_info.filename
                            break
                    
                    if txt_file_name:
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
            with open(path, 'r') as f:
                data_content = f.read()

        lines = data_content.splitlines()
        
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
        
        if current_levels > max_levels:
            max_levels = current_levels
        
        print(f"Found {sounding_count} soundings with maximum {max_levels} levels")
        
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
        
        sounding_idx = -1
        level_idx = 0
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('#'):
                sounding_idx += 1
                level_idx = 0
                
                year = int(line[13:17])
                month = int(line[18:20])
                day = int(line[21:23])
                hour = int(line[24:26])
                reltime = line[27:31].strip()
                numlev = int(line[32:36])
                p_src = line[37:45].strip()
                np_src = line[46:54].strip()
                
                dates[sounding_idx] = year * 10000 + month * 100 + day
                times[sounding_idx] = hour
                numlevs[sounding_idx] = numlev
                
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
                try:
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
                    
                    if lvltyp1_val and lvltyp1_val.isdigit():
                        lvltyp1[sounding_idx, level_idx] = float(lvltyp1_val)
                        
                    if lvltyp2_val and lvltyp2_val.isdigit():
                        lvltyp2[sounding_idx, level_idx] = float(lvltyp2_val)
                        
                    if etime_val:
                        etime[sounding_idx, level_idx] = float(etime_val)
                        
                    if pflag_val and pflag_val.isalnum():
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
                    continue


        stations_df = read_station_locations(save_file=False)

        if file_type.lower() in ['netcdf', 'nc']:
            if not download:
                temp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
                output_file = temp_file.name
                temp_file.close()
            else:
                output_file = os.path.join(download_dir, f"{file_name}-main.nc" if main else f"{file_name}-full.nc")

            with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
                ncfile.createDimension('num_profiles', sounding_count)
                ncfile.createDimension('levels', max_levels)
                
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

                date_var.units = 'YYYYMMDD'
                date_var.long_name = 'Date of sounding'
                
                time_var.units = 'HH'
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
                
                ncfile.description = f'IGRA sounding data for station {station_id}'
                ncfile.history = f'Created {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                ncfile.source = f'IGRA v2 data for {station_id}'

                try:
                    station_info = stations_df[stations_df['station_id'] == station_id].iloc[0]
                    ncfile.latitude = float(station_info['latitude'])
                    ncfile.longitude = float(station_info['longitude'])
                    ncfile.station_name = station_info['name']
                    ncfile.elevation = float(station_info['elevation'])
                except Exception as e:
                    print(f"Warning: Could not retrieve station location information: {e}")

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
            
            if not download:
                os.unlink(output_file)
                
            return ds
        
        elif file_type.lower() in ['pandas', 'df']:
            data = []
            
            try:
                station_info = stations_df[stations_df['station_id'] == station_id].iloc[0]
                station_lat = float(station_info['latitude'])
                station_lon = float(station_info['longitude'])
            except Exception as e:
                print(f"Warning: Could not retrieve station location information: {e}")
                station_lat = np.nan
                station_lon = np.nan
            
            for i in range(sounding_count):
                for j in range(max_levels):
                    if np.isnan(pressure[i, j]) and np.isnan(gph[i, j]) and np.isnan(temp[i, j]):
                        continue
                        
                    row = {
                        'num_profiles': i,
                        'date': dates[i],
                        'time': times[i],
                        'pressure': pressure[i, j],
                        'height': gph[i, j],
                        'temperature': temp[i, j],
                        'relative_humidity': rh[i, j],
                        'wind_direction': wdir[i, j],
                        'wind_speed': wspd[i, j],
                        'dewpoint_depression': dpdp[i, j],
                        'latitude': station_lat,
                        'longitude': station_lon
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
            
            df = pd.DataFrame(data)
            
            if download:
                df.to_csv(output_file, index=False)
                print(f"Successfully saved DataFrame to: {output_file}")
            
            return df
        
        else:
            raise ValueError("file_type must be either 'netcdf', 'pandas', or 'df'")
        
    except Exception as e:
        print(f"Error processing data for station {station_id}: {e}")
        return None
    
def _load_nc(file_path: Union[str, Path], print_info: bool = True) -> xr.Dataset:
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
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {file_path}")
    
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

def _load_df(file_path: Union[str, Path], print_info: bool = True) -> pd.DataFrame:
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
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
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

def load_data(file_path: Union[str, Path], print_info: bool = True) -> Union[pd.DataFrame, xr.Dataset]:
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
    >>> df = load_data("path/to/data.csv")
    
    >>> # Open a NetCDF file without printing information
    >>> ds = load_data("path/to/data.nc", print_info=False)
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.parquet', '.json', '.pkl']:
        return _load_df(file_path, print_info)
    elif file_path.suffix.lower() in ['.nc']:
        return _load_nc(file_path, print_info)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
def filter_by_date_range(df: Union[pd.DataFrame, xr.Dataset], 
                         start_date: int, 
                         end_date: int) -> Union[pd.DataFrame, xr.Dataset]:
    """Filter the data by date range.

    Parameters
    ----------
    df : Union[pd.DataFrame, xr.Dataset]
        Sounding data with 'date' column in YYYYMMDD format.
    start_date : int
        Start date in YYYYMMDD format.
    end_date : int
        End date in YYYYMMDD format.

    Returns
    -------
    Union[pd.DataFrame, xr.Dataset]
        Filtered data containing only data between start_date and end_date (inclusive).

    Examples
    --------
    >>> # Filter data for January 2020
    >>> filtered_data = filter_by_date_range(data, 20200101, 20200131)
    """
    if df is None:
        print("Error: No data provided")
        return None
        
    try:
        if isinstance(df, pd.DataFrame):
            filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if len(filtered_df) == 0:
                print(f"Warning: No data found between {start_date} and {end_date}")
            return filtered_df
            
        elif isinstance(df, xr.Dataset):
            date_mask = (df['date'].values >= start_date) & (df['date'].values <= end_date)
            if not np.any(date_mask):
                print(f"Warning: No data found between {start_date} and {end_date}")
                return None
                
            filtered_ds = df.isel(num_profiles=date_mask)
            return filtered_ds
            
        else:
            raise TypeError(f"Expected pandas DataFrame or xarray Dataset, got {type(df).__name__}")
            
    except Exception as e:
        print(f"Error filtering data: {e}")
        return None

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
            
        required_cols = ['num_profile', 'date', 'time']
        valid_vars = [var for var in variables if var in df.columns]
        cols_to_keep = required_cols + valid_vars
        
        filtered_df = df[cols_to_keep].copy()
        
        return filtered_df
    
    elif file_type == 'nc':
        if not isinstance(df, xr.Dataset):
            raise TypeError(f"Expected xarray Dataset, got {type(df).__name__}")
            
        required_vars = ['date', 'time']
        
        processed_vars = []
        for var in variables:
            if var == 'height':
                if 'gph' in df.variables:
                    processed_vars.append('gph')
            else:
                processed_vars.append(var)
                
        valid_vars = [var for var in processed_vars if var in df.variables]
        vars_to_keep = required_vars + valid_vars
        
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
        response = requests.get(metadata.IGRA_STATION_LIST_URL)
        response.raise_for_status()
        
        lines = response.text.splitlines()
        
        data_lines = [line for line in lines if line.strip() and not line.startswith('-')]
        
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
        
        df = pd.DataFrame(stations)
        
        numeric_cols = ['latitude', 'longitude', 'elevation', 'first_year', 'last_year', 'nobs']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
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

def plot_station_map(colour_by: str = 'none', 
                     year_range: Tuple[int, int] = (1900, 2025),
                     lat_range: Tuple[float, float] = (-90, 90),
                     lon_range: Tuple[float, float] = (-180, 180), 
                     stations: Optional[List[str]] = None,
                     last_updated_year: Optional[int] = None):
    """
    Displays an interactive map of IGRA stations using Plotly.
    
    Parameters
    ----------
    colour_by : str, optional
        Variable to use for colouring the stations. Options are:
        - 'none': No colouring (default)
        - 'elevation': Color stations by their elevation
        - 'last_year': Color stations by their last year of data
        - 'first_year': Color stations by their first year of data
        - 'nobs': Color stations by number of observations
    year_range : Tuple[int, int], optional
        Range of years to include in the map, by default (1900, 2025)
    lat_range : Tuple[float, float], optional
        Range of latitudes to display (min_lat, max_lat), by default (-90, 90)
    lon_range : Tuple[float, float], optional
        Range of longitudes to display (min_lon, max_lon), by default (-180, 180)
    stations : List[str], optional
        List of station IDs to display on the map, by default None
    last_updated_year : Optional[int], optional
        Last year of data availability to include in the map, by default None i.e., all stations with data availability from 1900 onwards
    Examples
    --------
    >>> # Display the map colored by elevation (default)
    >>> plot_station_map()
    
    >>> # Display the map colored by last year of data
    >>> plot_station_map(colour_by='last_year')
    
    >>> # Display the map colored by number of observations
    >>> plot_station_map(colour_by='nobs')
    
    >>> # Display stations in a specific region
    >>> plot_station_map(lat_range=(30, 50), lon_range=(-130, -70))  # North America
    """

    stations_df = read_station_locations(save_file=False)
    
    # Filter stations by year range
    stations_df = stations_df[
        (stations_df['first_year'] >= year_range[0]) & 
        (stations_df['last_year'] <= year_range[1])
    ]

    if last_updated_year is not None:
        stations_df = stations_df[stations_df['last_year'] >= last_updated_year]
    
    # Filter stations by latitude and longitude range
    stations_df = stations_df[
        (stations_df['latitude'] >= lat_range[0]) &
        (stations_df['latitude'] <= lat_range[1]) &
        (stations_df['longitude'] >= lon_range[0]) &
        (stations_df['longitude'] <= lon_range[1])
    ]

    if stations is not None:
        stations_df = stations_df[stations_df['station_id'].isin(stations)]

    if colour_by == 'none':
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

        if colour_by not in color_mapping:
            raise ValueError(f"colour_by must be one of {list(color_mapping.keys())}")

        color_info = color_mapping[colour_by]

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
        
        fig.update_traces(
            marker=dict(
                size=6,
                opacity=0.6,
                line=dict(width=0)
            )
        )

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
        ),
        coloraxis_colorbar=dict(
            title_font_size=36,
            tickfont_size=24
        )
    )
    
    fig.show()

def get_availability_json(station_id, download_dir=None, download_availability=False):
    """Get the availability of IGRA data for a given station.
    
    Parameters
    ----------
    station_id : str
        The station ID to fetch availability data for
    download_dir : Optional[str]
        Directory to save availability data. If None, uses current directory with date
    download_availability : bool, default=False
        Whether to save the availability data to a JSON file

    Returns
    -------
    Optional[Dict]
        Dictionary containing availability data with keys:
        - station_id: str
        - raw_data: List[List[int]] of [year, month, day, hour]
        - num_total_soundings: int
        - available_years: List[int]
        - num_soundings_per_year: Dict[int, int]
        - num_months_per_year: Dict[int, int]
        - num_days_per_year: Dict[int, int]
        Returns None if there was an error fetching the data

    Raises
    ------
    requests.exceptions.RequestException
        If there was an error downloading the data
    zipfile.BadZipFile
        If the downloaded file is not a valid zip file
    Exception
        For any other unexpected errors during processing

    Examples
    --------
    >>> # Get availability data for a station
    >>> availability = get_availability_json('USM00072518')
    >>> print(f"Station has {availability['num_total_soundings']} total soundings")
    
    >>> # Get and save availability data
    >>> availability = get_availability_json('USM00072518', 
    ...                                    download_dir='availability',
    ...                                    download_availability=True)
    """
    availability = []
    try:
        base_url = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/"
        zip_filename = f"{station_id}-data.txt.zip"
        zip_url = urljoin(base_url, zip_filename)
        
        try:
            print(f"Downloading data for station {station_id}...")
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            zip_buffer = io.BytesIO(response.content)
            
            print(f"Processing data for station {station_id}...")
            with zipfile.ZipFile(zip_buffer) as zip_ref:
                txt_file_name = None
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith('.txt'):
                        txt_file_name = file_info.filename
                        break
                
                if txt_file_name:
                    with zip_ref.open(txt_file_name) as source:
                        for line in source:
                            line = line.decode('utf-8').strip()
                            if line.startswith('#'):
                                year = int(line[13:17])
                                month = int(line[18:20])
                                day = int(line[21:23])
                                hour = int(line[24:26])
                                availability.append([year, month, day, hour])
                else:
                    raise Exception("No text file found in the zip archive")
            
            print(f"Successfully processed data for {station_id}")
        
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data for station {station_id}: {e}")
            return None
        except zipfile.BadZipFile:
            print(f"Error: The downloaded file for {station_id} is not a valid zip file.")
            return None
        except Exception as e:
            print(f"Unexpected error processing data for {station_id}: {e}")
            return None

        availability_grid = np.array(availability)
        years = np.unique(availability_grid[:, 0])

        year_soundings = {}
        for year, count in zip(np.unique(availability_grid[:, 0]), np.bincount(availability_grid[:, 0], minlength=int(np.max(availability_grid[:, 0])+1))[int(np.min(availability_grid[:, 0])):]):
            year_soundings[int(year)] = int(count)

        year_months = {}
        for year in np.unique(availability_grid[:, 0]):
            year_data = availability_grid[availability_grid[:, 0] == year]
            unique_months = np.unique(year_data[:, 1])
            month_count = len(unique_months)
            year_months[int(year)] = int(month_count)

        year_days = {}
        for year in np.unique(availability_grid[:, 0]):
            year_data = availability_grid[availability_grid[:, 0] == year]
            day_of_year = year_data[:, 1] * 100 + year_data[:, 2]
            unique_days = np.unique(day_of_year)
            day_count = len(unique_days)
            year_days[int(year)] = int(day_count)

        availability_data = {
            'station_id': station_id,
            'raw_data': availability,
            'num_total_soundings': len(availability),
            'available_years': years.tolist(),
            'num_soundings_per_year': year_soundings,
            'num_months_per_year': year_months,
            'num_days_per_year': year_days
        }

        if download_availability:
            if download_dir is None:
                download_dir = os.path.join(os.getcwd(), str(datetime.datetime.now().strftime("%Y-%m-%d")))

            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            availability_file = os.path.join(download_dir, f"{station_id}-availability.json")
            with open(availability_file, 'w') as f:
                json.dump(availability_data, f, indent=4)
            print(f"Availability data saved to {availability_file}")
            
        return availability_data
    
    except Exception as e:
        print(f"Error fetching availability for station {station_id}: {e}")
        return None

def filter_stations(start_year: Optional[int] = None, 
                     end_year: Optional[int] = None, 
                     lat_range: Optional[Tuple[float, float]] = None,
                     lon_range: Optional[Tuple[float, float]] = None,
                     has_date_range: Optional[Tuple[str, str]] = None,
                     availability_dir: Optional[str] = None) -> List[str]:
    """Filter station data by year, latitude, and longitude range.
    
    Parameters
    ----------
    start_year : Optional[int]
        First year of data availability to include
    end_year : Optional[int]
        Last year of data availability to include
    lat_range : Optional[Tuple[float, float]]
        Range of latitudes to include (min_lat, max_lat)
    lon_range : Optional[Tuple[float, float]]
        Range of longitudes to include (min_lon, max_lon)
    has_date_range : Optional[Tuple[int, int]]
        Contains records between start_date and end_date
    availability_dir : Optional[str]
        Directory containing availability data

    Returns
    -------
    List[str]
        List of station IDs that meet the filtering criteria
        
    Raises
    ------
    ValueError
        If no filtering criteria are provided
        
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
    if all(param is None for param in [start_year, end_year, lat_range, lon_range]):
        raise ValueError("At least one filtering criterion must be provided")
        
    stations_df = read_station_locations(save_file=False)
    
    if start_year is not None or end_year is not None:
        if start_year is not None:
            stations_df = stations_df[stations_df['first_year'] >= start_year]
        if end_year is not None:
            stations_df = stations_df[stations_df['last_year'] <= end_year]
    
    if lat_range is not None:
        stations_df = stations_df[
            (stations_df['latitude'] >= lat_range[0]) &
            (stations_df['latitude'] <= lat_range[1])
        ]
    if lon_range is not None:
        stations_df = stations_df[
            (stations_df['longitude'] >= lon_range[0]) &
            (stations_df['longitude'] <= lon_range[1])
        ]

    stations_list = stations_df['station_id'].tolist()
    
    if has_date_range is not None:
        start_date = has_date_range[0]
        end_date = has_date_range[1]
        for station_id in stations_list[:]:
            if availability_dir is None:
                availability_data = get_availability_json(station_id)
            else:
                availability_file = os.path.join(availability_dir, f"{station_id}-availability.json")
                with open(availability_file, 'r') as f:
                    availability_data = json.load(f)
                
            has_data = False
            for year, month, day, hour in availability_data['raw_data']:
                current_date = year * 10000 + month * 100 + day
                
                if start_date <= current_date <= end_date:
                    has_data = True
                    break
                    
            if not has_data:
                stations_list.remove(station_id)

    return stations_list

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
    invalid_values = [np.nan, -9999, -8888, 9999, 8888]
    
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
                        method: str = 'linear',
                        start_date: Optional[int] = None,
                        end_date: Optional[int] = None,
                        fill_value: Optional[float] = None,
                        **kwargs) -> Union[pd.DataFrame, xr.Dataset]:
    """Interpolate station data onto a uniform grid using scipy's interpolation functions.

    This function creates a uniform grid of the index variable and interpolates
    the specified variable onto that grid using scipy's interpolation functions.
    It supports both DataFrame and NetCDF data formats.

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
    method : str, optional
        Interpolation method to use. See scipy.interpolate.interp1d for available methods.
        Default is 'linear'.
    start_date : Optional[int], optional
        Start date for filtering data, by default None
    end_date : Optional[int], optional
        End date for filtering data, by default None
    fill_value : Optional[float], optional
        Value to use for extrapolation, by default None
    **kwargs : dict
        Additional keyword arguments passed to scipy's interpolation functions.
        See scipy.interpolate.interp1d documentation for details.

    Returns
    -------
    Union[pd.DataFrame, xr.Dataset]
        Interpolated data with the same format as the input, but with uniform grid spacing
    """
    from scipy import interpolate
    import numpy as np

    if start_date is not None or end_date is not None:
        if isinstance(data, pd.DataFrame):
            data = filter_by_date_range(data, start_date, end_date, file_type='df')
        else:
            data = filter_by_date_range(data, start_date, end_date, file_type='nc')
    
    grid = np.arange(min_index, max_index + step_size, step_size)
            
    if isinstance(data, pd.DataFrame):
        if variable not in data.columns:
            raise ValueError(f"Variable '{variable}' not found in DataFrame")
            
        if index_variable not in data.columns:
            raise ValueError(f"Index variable '{index_variable}' not found in DataFrame")
            
        interpolated_data = data.copy()
        
        grouped = interpolated_data.groupby('num_profiles')
        
        interpolated_profiles = []
        skipped_profiles = []
        
        for profile_num, profile in grouped:
            x = profile[index_variable].values
            y = profile[variable].values
            
            x, y = _filter_invalid_values(x, y)
            
            if len(x) > 1:
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = y[sort_idx]
                
                try:
                    f = interpolate.interp1d(x, y, kind=method, bounds_error=False, fill_value=fill_value, **kwargs)
                    interpolated_values = f(grid)
                    
                    profile_df = pd.DataFrame({
                        'num_profiles': profile_num,
                        'date': profile['date'].iloc[0],
                        'time': profile['time'].iloc[0],
                        index_variable: grid,
                        variable: interpolated_values
                    })
                    
                    interpolated_profiles.append(profile_df)
                except ValueError as e:
                    skipped_profiles.append((profile_num, profile['date'].iloc[0], profile['time'].iloc[0], str(e)))
            else:
                skipped_profiles.append((profile_num, profile['date'].iloc[0], profile['time'].iloc[0], "Insufficient valid data points"))
        
        if skipped_profiles:
            print("\nThe following profiles were skipped during interpolation:")
            for profile_num, date, time, reason in skipped_profiles:
                print(f"Profile {profile_num} (Date: {date}, Time: {time}): {reason}")
            print()
        
        if interpolated_profiles:
            return pd.concat(interpolated_profiles, ignore_index=True)
        else:
            return pd.DataFrame()
        
    elif isinstance(data, (xr.Dataset, nc.Dataset)):
        if isinstance(data, nc.Dataset):
            data = xr.Dataset.from_dict(data.variables)
            
        var_name = 'gph' if variable == 'height' else variable
        index_name = 'gph' if index_variable == 'height' else index_variable
        
        if var_name not in data.variables:
            raise ValueError(f"Variable '{variable}' not found in Dataset")
            
        if index_name not in data.variables:
            raise ValueError(f"Index variable '{index_variable}' not found in Dataset")
            
        interpolated_data = data.copy()
        
        new_coord = xr.DataArray(
            grid,
            dims=['levels'],
            coords={'levels': grid}
        )
        
        interpolated_values = []
        skipped_profiles = []
        
        for profile_idx in range(len(interpolated_data.num_profiles)):
            x = interpolated_data[index_name].isel(num_profiles=profile_idx).values
            y = interpolated_data[var_name].isel(num_profiles=profile_idx).values
            
            x, y = _filter_invalid_values(x, y)
            
            if len(x) > 1:  # Need at least 2 points for interpolation
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = y[sort_idx]
                
                try:
                    f = interpolate.interp1d(x, y, kind=method, bounds_error=False, fill_value=fill_value, **kwargs)
                    interpolated_values.append(f(grid))
                except ValueError as e:
                    date = interpolated_data['date'].isel(num_profiles=profile_idx).values
                    time = interpolated_data['time'].isel(num_profiles=profile_idx).values
                    skipped_profiles.append((profile_idx, date, time, str(e)))
                    interpolated_values.append(np.full_like(grid, np.nan))
            else:
                date = interpolated_data['date'].isel(num_profiles=profile_idx).values
                time = interpolated_data['time'].isel(num_profiles=profile_idx).values
                skipped_profiles.append((profile_idx, date, time, "Insufficient valid data points"))
                interpolated_values.append(np.full_like(grid, np.nan))
        
        if skipped_profiles:
            print("\nThe following profiles were skipped during interpolation:")
            for profile_idx, date, time, reason in skipped_profiles:
                print(f"Profile {profile_idx} (Date: {date}, Time: {time}): {reason}")
            print()
        
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
        
        new_data['date'] = interpolated_data['date']
        new_data['time'] = interpolated_data['time']
        
        return new_data
    
    else:
        raise TypeError(f"Expected pandas DataFrame, xarray Dataset, or netCDF4 Dataset, got {type(data).__name__}")

def interp_data_to_pressure_levels(data: Union[pd.DataFrame, xr.Dataset, nc.Dataset], 
                               variable: str,
                               start_date: Optional[int] = None,
                               end_date: Optional[int] = None,
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
    start_date : Optional[int]
        Start date for filtering data, by default None
    end_date : Optional[int]
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
    if start_date is not None or end_date is not None:
        if isinstance(data, pd.DataFrame):
            data = filter_by_date_range(data, start_date, end_date, file_type='df')
        else:
            data = filter_by_date_range(data, start_date, end_date, file_type='nc')
    
    grid = [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
    log_grid = np.log(grid)
            
    if isinstance(data, pd.DataFrame):
        if variable not in data.columns:
            raise ValueError(f"Variable '{variable}' not found in DataFrame")
            
        interpolated_data = data.copy()
        
        grouped = interpolated_data.groupby('num_profiles')
        
        interpolated_profiles = []
        
        for profile_num, profile in grouped:
            x = profile['pressure'].values
            y = profile[variable].values
            
            x, y = _filter_invalid_values(x, y)
            log_x = np.log(x)
            sort_idx = np.argsort(log_x)
            log_x = log_x[sort_idx]
            y = y[sort_idx]

            
            if len(x) > 1:  # Need at least 2 points for interpolation
                interpolated_values = np.interp(log_grid, log_x, y, left=fill_value, right=fill_value)
                
                profile_df = pd.DataFrame({
                    'num_profiles': profile_num,
                    'date': profile['date'].iloc[0],
                    'time': profile['time'].iloc[0],
                    'pressure': grid,
                    variable: interpolated_values
                })
                
                interpolated_profiles.append(profile_df)
        
        if interpolated_profiles:
            return pd.concat(interpolated_profiles, ignore_index=True)
        else:
            return pd.DataFrame()
        
    elif isinstance(data, (xr.Dataset, nc.Dataset)):
        if isinstance(data, nc.Dataset):
            data = xr.Dataset.from_dict(data.variables)
            
        var_name = 'gph' if variable == 'height' else variable
        
        if var_name not in data.variables:
            raise ValueError(f"Variable '{variable}' not found in Dataset")
            
        interpolated_data = data.copy()
        
        new_coord = xr.DataArray(
            grid,
            dims=['levels'],
            coords={'levels': grid}
        )
        
        interpolated_values = []
        for profile_idx in range(len(interpolated_data.num_profiles)):
            x = interpolated_data['pressure'].isel(num_profiles=profile_idx).values
            y = interpolated_data[var_name].isel(num_profiles=profile_idx).values
            
            x, y = _filter_invalid_values(x, y)
            log_x = np.log(x)
            sort_idx = np.argsort(log_x)
            log_x = log_x[sort_idx]
            y = y[sort_idx]
            
            if len(x) > 1:
                interpolated_values.append(np.interp(log_grid, log_x, y, left=fill_value, right=fill_value))
            else:
                interpolated_values.append(np.full_like(log_grid, np.nan))
        
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
        
        new_data['date'] = interpolated_data['date']
        new_data['time'] = interpolated_data['time']
        
        return new_data
    
    else:
        raise TypeError(f"Expected pandas DataFrame, xarray Dataset, or netCDF4 Dataset, got {type(data).__name__}")

def get_availability(data: Union[pd.DataFrame, xr.Dataset]) -> Optional[Dict]:
    """Get the availability of IGRA data from a DataFrame or Dataset.
    
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
                    day: [hour, ...]
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
    if data is None:
        print("Error: No data provided")
        return None
        
    try:
        availability = []
        
        if isinstance(data, pd.DataFrame):
            for _, row in data.groupby('num_profiles').first().iterrows():
                date = int(float(row['date'])) if isinstance(row['date'], (str, float)) else int(row['date'])
                time = int(float(row['time'])) if isinstance(row['time'], (str, float)) else int(row['time'])
                
                year = date // 10000
                month = (date % 10000) // 100
                day = date % 100
                
                availability.append([year, month, day, time])
                
        elif isinstance(data, (xr.Dataset, nc.Dataset)):
            dates = data['date'].values
            times = data['time'].values
            
            for date, time in zip(dates, times):
                date = int(float(date)) if isinstance(date, (str, float)) else int(date)
                time = int(float(time)) if isinstance(time, (str, float)) else int(time)
                
                year = date // 10000
                month = (date % 10000) // 100
                day = date % 100
                
                availability.append([year, month, day, time])
                
        else:
            raise TypeError(f"Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")

        nested_availability = {}
        
        availability.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        
        for year, month, day, time in availability:
            if year not in nested_availability:
                nested_availability[year] = {}
            
            if month not in nested_availability[year]:
                nested_availability[year][month] = {}
            
            if day not in nested_availability[year][month]:
                nested_availability[year][month][day] = []
            
            if time not in nested_availability[year][month][day]:
                nested_availability[year][month][day].append(time)
                nested_availability[year][month][day].sort()

        return nested_availability
        
    except Exception as e:
        print(f"Error processing availability data: {e}")
        return None

def get_years(data: Union[pd.DataFrame, xr.Dataset]) -> List[int]:
    """Get the years of the data."""
    availability = get_availability(data)
    if availability is None:
        return []
    return list(availability.keys())

def get_months(data: Union[pd.DataFrame, xr.Dataset], year: int) -> List[int]:
    """Get the months of the data for a specific year."""
    availability = get_availability(data)
    if availability is None:
        return []
    return list(availability[year].keys())

def get_days(data: Union[pd.DataFrame, xr.Dataset], year: int, month: int) -> List[int]:
    """Get the days of the data for a specific year and month."""
    availability = get_availability(data)
    if availability is None:
        return []
    return list(availability[year][month].keys())

def get_times(data: Union[pd.DataFrame, xr.Dataset], year: int, month: int, day: int) -> List[str]:
    """Get the times of the data for a specific date."""
    availability = get_availability(data)
    if availability is None:
        return []
    return availability[year][month][day]

def get_num_soundings(data: Union[pd.DataFrame, xr.Dataset]) -> int:
    """Get the number of soundings in the data."""
    availability = get_availability(data)
    if availability is None:
        return 0
    total = 0
    for year in availability:
        for month in availability[year]:
            for day in availability[year][month]:
                total += len(availability[year][month][day])
    return total

def plot_profile(data: Union[pd.DataFrame, xr.Dataset],
                x_variable: str,
                y_variable: str,
                date: Optional[int] = None,  # Format: YYYYMMDD
                time: Optional[int] = None,  # Format: HH
                figsize: Tuple[int, int] = (10, 8),
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                grid: bool = True,
                show: bool = True) -> Optional[plt.Figure]:
    """Plot a vertical profile for a specific date and time.
    
    If date and time are not provided, the function will use the single unique date/time
    if it exists in the data. Otherwise, an error will be raised.
    """
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

    if data is None:
        print("Error: No data provided")
        return None
        
    is_single_profile = False
    if isinstance(data, (xr.Dataset, nc.Dataset)):
        if 'num_profiles' in data.dims and data.dims['num_profiles'] == 1:
            is_single_profile = True
            date = data['date'].values[0]
            time = data['time'].values[0]

    if isinstance(data, pd.DataFrame):
        unique_dates = data['date'].unique()
        unique_times = data['time'].unique()
    else:
        dates = data['date'].values
        times = data['time'].values
        
        unique_dates = np.unique(dates)
        unique_times = np.unique(times)

    if (date is None or time is None) and not is_single_profile:
        if len(unique_dates) == 1 and len(unique_times) == 1:
            date = unique_dates[0]
            time = unique_times[0]
        else:
            raise ValueError("Multiple dates/times found in data. Please specify date and time.")

    if isinstance(data, pd.DataFrame):
        profile_data = data[
            (data['date'] == date) &
            (data['time'] == time)
        ]

        if len(profile_data) == 0:
            print(f"No profile found for {date} {time}")
            return None

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

    elif isinstance(data, (xr.Dataset, nc.Dataset)):
        if is_single_profile:
            profile_data = data
        else:
            target_idx = np.where((data['date'].values == date) & 
                                (data['time'].values == time))[0]
            
            if len(target_idx) == 0:
                print(f"No profile found for {date} {time}")
                return None
                
            profile_data = data.isel(num_profiles=target_idx[0])
        
        x_variable = 'gph' if x_variable == 'height' else x_variable
        y_variable = 'gph' if y_variable == 'height' else y_variable
        
        x_array = profile_data[x_variable]
        y_array = profile_data[y_variable]
        
        x_values = x_array.values.flatten()
        y_values = y_array.values.flatten()
        
        valid_mask = (
            ~np.isnan(x_values) & 
            ~np.isnan(y_values) &
            (x_values != -9999) &
            (x_values != -8888) &
            (y_values != -9999) &
            (y_values != -8888)
        )
        
        x_data = x_values[valid_mask]
        y_data = y_values[valid_mask]

    else:
        print(f"Error: Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")
        return None

    if len(x_data) == 0 or len(y_data) == 0:
        print(f"Error: No valid data points found for {x_variable} vs {y_variable}")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x_data, y_data)
    
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
        ax.set_title(f"{y_variable.capitalize()} vs {x_variable.capitalize()}\n{date} {time}")
        
    if y_variable == 'pressure':
        ax.invert_yaxis()
        
    if grid:
        ax.grid(True)
        
    if show:
        plt.show()
        
    return fig

def get_profile(data: Union[pd.DataFrame, xr.Dataset],
                date: str,
                time: str) -> Optional[Union[pd.DataFrame, xr.Dataset]]:
    """Get a profile for a specific date and time.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        Input data containing the profiles
    date : str
        Date in YYYYMMDD format
    time : str
        Time in HH format
        
    Returns
    -------
    Optional[Union[pd.DataFrame, xr.Dataset]]
        The profile data if found, None otherwise
    """
    if data is None:
        print("Error: No data provided")
        return None

    if isinstance(data, pd.DataFrame):
        profile_data = data[
            (data['date'] == date) &
            (data['time'] == time)
        ]
        
        if len(profile_data) == 0:
            profile_data = data[
                (data['date'] == date) &
                (data['time'] == time)
            ]

        if len(profile_data) == 0:
            print(f"No profile found for {date} {time}")
            return None

        return profile_data
        
    elif isinstance(data, (xr.Dataset, nc.Dataset)):
        if isinstance(data, nc.Dataset):
            data = xr.Dataset.from_dict(data.variables)
            
        if 'date' not in data.variables or 'time' not in data.variables:
            print("Error: Dataset must contain 'date' and 'time' variables")
            return None
            
        dates = data['date'].values
        times = data['time'].values
        
        target_idx = np.where((dates == date) & 
                            (times == time))[0]
        
        if len(target_idx) == 0:
            print(f"No profile found for {date} {time}")
            return None
            
        new_data = xr.Dataset()
        
        for var_name in data.variables:
            if 'num_profiles' in data[var_name].dims:
                new_data[var_name] = xr.DataArray(
                    data[var_name].values[target_idx[0]:target_idx[0]+1],
                    dims=['num_profiles'] + [d for d in data[var_name].dims if d != 'num_profiles'],
                    coords={'num_profiles': [0]},
                    attrs=data[var_name].attrs
                )
            else:
                new_data[var_name] = data[var_name]
                
        new_data.attrs.update(data.attrs)
        
        return new_data
            
    else:
        print(f"Error: Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")
        return None
    
def get_availability_json_batch(directory: str, station_list: Optional[List[str]] = None):
    """Download availability data for all stations or a list of stations.
    
    Parameters
    ----------
    directory : str
        Directory to save availability data
    station_list : Optional[List[str]]
        List of station IDs to process. If None, processes all stations in the IGRA database

    Returns
    -------
    None

    Raises
    ------
    Exception
        For any unexpected errors during processing

    Examples
    --------
    >>> # Download availability data for all stations
    >>> download_availability('availability')
    
    >>> # Download availability data for specific stations
    >>> download_availability('availability', ['USM00072518', 'USM00072456'])
    """
    if station_list is None:
        stations_df = read_station_locations(save_file=False)
        station_list = list(stations_df['station_id'])
        
    error_stations = []
    for i,station_id in enumerate(station_list):
        try:
            availability_data = get_availability_json(station_id, download_dir=directory, download_availability=True)
        except Exception as e:
            error_stations.append(station_id)
            print(f"Error processing station {station_id}: {e}")
            continue
        print(f"Station {i+1} of {len(station_list)}: {station_id} has {availability_data['num_total_soundings']} soundings")

    print(f"Error processing {len(error_stations)} stations: {error_stations}")

def save_data(data: Union[pd.DataFrame, xr.Dataset], name: str):
    """Save a Dataset to a netcdf file or a DataFrame to a csv file.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        Data to save
    name : str
        Name of the file to save
    """
    if isinstance(data, xr.Dataset):
        data.to_netcdf(name)
    elif isinstance(data, pd.DataFrame):
        data.to_csv(name, index=False)
    else:
        print(f"Error: Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")
        return None

def convert_to_netcdf(data: Union[pd.DataFrame, xr.Dataset], name: str):
    """Convert data to netCDF format and save it.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        Data to convert and save
    name : str
        Name of the output netCDF file
        
    Returns
    -------
    None
    """
    if isinstance(data, xr.Dataset):
        raise ValueError("Data is already in netCDF format.")
    elif isinstance(data, pd.DataFrame):
        datetime_cols = []
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                datetime_cols.append(col)
        
        df = data.copy()
        
        for col in datetime_cols:
            if col == 'date':
                df[col] = df[col].dt.strftime('%Y%m%d').astype(np.int32)
            elif col == 'time':
                df[col] = df[col].dt.hour.astype(np.int32)
        
        profiles = df.groupby(['date', 'time'])
        
        num_profiles = len(profiles)
        max_levels = profiles.size().max()
        
        dates = np.zeros(num_profiles, dtype=np.int32)
        times = np.zeros(num_profiles, dtype=np.int32)
        
        pressure = np.full((num_profiles, max_levels), np.nan, dtype=np.float32)
        gph = np.full((num_profiles, max_levels), np.nan, dtype=np.float32)
        temp = np.full((num_profiles, max_levels), np.nan, dtype=np.float32)
        rh = np.full((num_profiles, max_levels), np.nan, dtype=np.float32)
        wind_dir = np.full((num_profiles, max_levels), np.nan, dtype=np.float32)
        wind_speed = np.full((num_profiles, max_levels), np.nan, dtype=np.float32)
        dewpoint = np.full((num_profiles, max_levels), np.nan, dtype=np.float32)
        
        for i, ((date, time), profile) in enumerate(profiles):
            dates[i] = date
            times[i] = time
            
            profile_data = profile.reset_index(drop=True)
            
            if 'pressure' in profile_data.columns:
                pressure[i, :len(profile_data)] = profile_data['pressure'].values
            if 'gph' in profile_data.columns:
                gph[i, :len(profile_data)] = profile_data['gph'].values
            elif 'height' in profile_data.columns:
                gph[i, :len(profile_data)] = profile_data['height'].values
            if 'temperature' in profile_data.columns:
                temp[i, :len(profile_data)] = profile_data['temperature'].values
            if 'relative_humidity' in profile_data.columns:
                rh[i, :len(profile_data)] = profile_data['relative_humidity'].values
            if 'wind_direction' in profile_data.columns:
                wind_dir[i, :len(profile_data)] = profile_data['wind_direction'].values
            if 'wind_speed' in profile_data.columns:
                wind_speed[i, :len(profile_data)] = profile_data['wind_speed'].values
            if 'dewpoint' in profile_data.columns:
                dewpoint[i, :len(profile_data)] = profile_data['dewpoint'].values
        
        ds = xr.Dataset(
            data_vars={
                'date': (['num_profiles'], dates),
                'time': (['num_profiles'], times),
                'pressure': (['num_profiles', 'levels'], pressure),
                'gph': (['num_profiles', 'levels'], gph),
                'temperature': (['num_profiles', 'levels'], temp),
                'relative_humidity': (['num_profiles', 'levels'], rh),
                'wind_direction': (['num_profiles', 'levels'], wind_dir),
                'wind_speed': (['num_profiles', 'levels'], wind_speed),
                'dewpoint': (['num_profiles', 'levels'], dewpoint)
            },
            coords={
                'num_profiles': np.arange(num_profiles),
                'levels': np.arange(max_levels)
            }
        )
        
        ds.attrs['title'] = 'IGRA Data'
        ds.attrs['source'] = 'Converted from DataFrame'
        ds.attrs['creation_date'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        ds.attrs['latitude'] = df['latitude'].iloc[0]
        ds.attrs['longitude'] = df['longitude'].iloc[0]

        
        ds.pressure.attrs['units'] = 'hPa'
        ds.gph.attrs['units'] = 'm'
        ds.temperature.attrs['units'] = '°C'
        ds.relative_humidity.attrs['units'] = '%'
        ds.wind_direction.attrs['units'] = 'degrees'
        ds.wind_speed.attrs['units'] = 'm/s'
        ds.dewpoint.attrs['units'] = '°C'
        
        ds.to_netcdf(name)
    else:
        print(f"Error: Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")
        return None

def convert_to_df(data: Union[pd.DataFrame, xr.Dataset], name: str):
    """Convert data to CSV format and save it.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        Data to convert and save
    name : str
        Name of the output CSV file
        
    Returns
    -------
    None
    """
    if isinstance(data, pd.DataFrame):
        raise ValueError("Data is already in DataFrame format.")
    elif isinstance(data, xr.Dataset):
        df = data.to_dataframe()
        
        df = df.reset_index()
        
        if 'levels' in df.columns:
            df = df.drop(columns=['levels'])
        
        df.to_csv(name, index=False)
    else:
        print(f"Error: Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")
        return None
    
def compute_potential_temperature(data: Union[pd.DataFrame, xr.Dataset, nc.Dataset]) -> Union[pd.DataFrame, xr.Dataset]:
    """Compute potential temperature for every observation in the data.
    
    Potential temperature is calculated using the formula:
    θ = T * (1000/p)^(2/7)
    where θ is potential temperature in Kelvin, T is temperature in Kelvin, and p is pressure in hPa.
    
    The function first converts temperature from Celsius to Kelvin, then applies the potential
    temperature formula. Invalid values (NaN, -9999, -8888) are preserved as NaN in the output.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset, nc.Dataset]
        Input data containing temperature and pressure information. Can be:
        - pandas DataFrame with 'temperature' and 'pressure' columns
        - xarray Dataset with 'temperature' and 'pressure' variables
        - netCDF4 Dataset with 'temperature' and 'pressure' variables
        
    Returns
    -------
    Union[pd.DataFrame, xr.Dataset]
        Data with the same structure as input plus a new 'potential_temperature' variable/column.
        For DataFrames: adds 'potential_temperature' column
        For Datasets: adds 'potential_temperature' variable with same dimensions as 'temperature'
        
    Examples
    --------
    >>> # Compute potential temperature for DataFrame
    >>> df = read_station_data("USM00072520", file_type='df')
    >>> df_with_theta = compute_potential_temperature(df)
    >>> print(df_with_theta.columns)
    ['num_profiles', 'date', 'time', 'pressure', 'height', 'temperature', 
     'relative_humidity', 'wind_direction', 'wind_speed', 'dewpoint_depression',
     'latitude', 'longitude', 'potential_temperature']
    
    >>> # Compute potential temperature for NetCDF Dataset
    >>> ds = read_station_data("USM00072520", file_type='netcdf')
    >>> ds_with_theta = compute_potential_temperature(ds)
    >>> print(list(ds_with_theta.variables))
    ['date', 'time', 'pressure', 'gph', 'temperature', 'rh', 'wdir', 'wspd', 
     'dpdp', 'potential_temperature']
    
    >>> # Access potential temperature values
    >>> theta_values = ds_with_theta['potential_temperature'].values
    >>> print(f"Potential temperature range: {np.nanmin(theta_values):.1f} - {np.nanmax(theta_values):.1f} K")
    """
    
    if data is None:
        print("Error: No data provided")
        return None
        
    try:
        if isinstance(data, pd.DataFrame):
            if 'temperature' not in data.columns:
                raise ValueError("DataFrame must contain 'temperature' column")
            if 'pressure' not in data.columns:
                raise ValueError("DataFrame must contain 'pressure' column")
                
            result_data = data.copy()
            
            temp_celsius = result_data['temperature'].copy()
            pressure_hpa = result_data['pressure'].copy()
            
            valid_mask = (
                ~temp_celsius.isna() & 
                ~pressure_hpa.isna() &
                (temp_celsius != -9999) & 
                (temp_celsius != -8888) &
                (pressure_hpa != -9999) & 
                (pressure_hpa != -8888) &
                (pressure_hpa > 0)
            )
            
            potential_temp = np.full(len(result_data), np.nan)
            
            temp_kelvin = temp_celsius[valid_mask] + 273.15
            p_hpa = pressure_hpa[valid_mask]
            
            potential_temp[valid_mask] = temp_kelvin * (1000 / p_hpa) ** (2/7)
            
            result_data['potential_temperature'] = potential_temp
            
            return result_data
            
        elif isinstance(data, (xr.Dataset, nc.Dataset)):
            if isinstance(data, nc.Dataset):
                data = xr.Dataset.from_dict(data.variables)
                
            if 'temperature' not in data.variables:
                raise ValueError("Dataset must contain 'temperature' variable")
            if 'pressure' not in data.variables:
                raise ValueError("Dataset must contain 'pressure' variable")
                
            result_data = data.copy()
            
            temp_celsius = result_data['temperature'].values
            pressure_hpa = result_data['pressure'].values
            
            valid_mask = (
                ~np.isnan(temp_celsius) & 
                ~np.isnan(pressure_hpa) &
                (temp_celsius != -9999) & 
                (temp_celsius != -8888) &
                (pressure_hpa != -9999) & 
                (pressure_hpa != -8888) &
                (pressure_hpa > 0)  # Pressure must be positive
            )
            
            potential_temp = np.full_like(temp_celsius, np.nan)
            
            temp_kelvin = temp_celsius[valid_mask] + 273.15
            p_hpa = pressure_hpa[valid_mask]
            
            potential_temp[valid_mask] = temp_kelvin * (1000 / p_hpa) ** (2/7)
            
            potential_temp_da = xr.DataArray(
                potential_temp,
                dims=result_data['temperature'].dims,
                coords=result_data['temperature'].coords,
                attrs={
                    'units': 'K',
                    'long_name': 'Potential temperature',
                    'description': 'Potential temperature calculated as θ = T * (1000/p)^(2/7)'
                }
            )
            
            result_data['potential_temperature'] = potential_temp_da
            
            return result_data
            
        else:
            raise TypeError(f"Expected pandas DataFrame or xarray Dataset, got {type(data).__name__}")
            
    except Exception as e:
        print(f"Error computing potential temperature: {e}")
        return None


#### DATE FORMAT YYYYMMDD
#### TIME FORMAT HH

