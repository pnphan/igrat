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
from datetime import datetime
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
import igrat

stations_df = igrat.read_station_locations()

station_ids = list(stations_df['station_id'])

def get_availability_json(station_id, download=True, download_dir=None, download_availability=True):
    """
    Get the availability of IGRA data for a given station.
    """
    availability = []
    try:
        if download_dir is None:
            download_dir = os.path.join(os.getcwd(), str(datetime.now().strftime("%Y-%m-%d")))

        # Create the directory if it doesn't exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        if download:
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
            availability_file = os.path.join(download_dir, f"{station_id}-availability.json")
            with open(availability_file, 'w') as f:
                json.dump(availability_data, f, indent=4)
            print(f"Availability data saved to {availability_file}")
            
        return availability_data
    
    except Exception as e:
        print(f"Error fetching availability for station {station_id}: {e}")
        return None


if __name__ == "__main__":
    error_stations = []
    for i,station_id in enumerate(station_ids):
        try:
            availability_data = get_availability_json(station_id, download_dir='availability')
        except Exception as e:
            error_stations.append(station_id)
            print(f"Error processing station {station_id}: {e}")
            continue
        print(f"Station {i+1} of {len(station_ids)}: {station_id} has {availability_data['num_total_soundings']} soundings")

    print(f"Error processing {len(error_stations)} stations: {error_stations}")

# June 6, 2025 18:38:00 EST