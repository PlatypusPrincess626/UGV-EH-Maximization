import numpy as np
import requests
import pandas as pd


def fetch_topography_from_usgs(lat_center, lon_center, grid_size=800, step_deg=0.0003):
    """Fetches elevation grid using USGS EPQS API."""
    elevation_grid = np.zeros((grid_size, grid_size))
    start_lat = lat_center + (grid_size / 2) * step_deg
    start_lon = lon_center - (grid_size / 2) * step_deg
    url = "https://epqs.nationalmap.gov/v1/json"

    print("Fetching topography from USGS (this may take time)...")
    for y in range(grid_size):
        for x in range(grid_size):
            lat = start_lat - (y * step_deg)
            lon = start_lon + (x * step_deg)
            params = {'x': lon, 'y': lat, 'units': 'Meters', 'output': 'json'}
            try:
                response = requests.get(url, params=params)
                elevation_grid[y, x] = response.json()['value']
            except Exception:
                elevation_grid[y, x] = 0
        if y % 100 == 0:
            print(f"Progress: {y / grid_size * 100:.1f}%")
    return elevation_grid


if __name__ == "__main__":
    # Settings from your original environment.py
    LAT, LON, SIZE, STEP = 44.424, -110.589, 800, 0.000009

    topo_data = fetch_topography_from_usgs(LAT, LON, SIZE, STEP)

    # Save as CSV
    pd.DataFrame(topo_data).to_csv("yellowstone_topo.csv", index=False, header=False)
    print("Topography saved to yellowstone_topo.csv")