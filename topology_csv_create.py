import requests
import rasterio
import math

center_lat = 44.424
center_lon = -110.589

side = 800 + math.ceil(50.0 / math.tan(math.radians(12)))  # 1036 m
half = side / 2.0

meters_per_deg_lat = 111320.0
meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))

lat_offset = half / meters_per_deg_lat
lon_offset = half / meters_per_deg_lon

# 1. Fetch the data once using your bounding box
url = "https://portal.opentopography.org/API/globaldem"
params = {
    'demtype': 'SRTMGL1',
    'south': center_lat - lat_offset,
    'north': center_lat + lat_offset,
    'west': center_lon - lon_offset,
    'east': center_lon + lon_offset,
    'outputFormat': 'GTiff',
    'API_Key': '7c40b10aed445fd24fa9a7b13286e64f'
}
response = requests.get(url, params=params)

if response.status_code == 200:
    # Save the binary content correctly
    output_path = 'topo_data.tif'
    with open(output_path, 'wb') as f:
        f.write(response.content)

    print(f"File saved successfully to {output_path}")

    # Now verify with rasterio
    try:
        with rasterio.open(output_path) as src:
            print("Successfully opened TIFF!")
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"CRS: {src.crs}")
    except Exception as e:
        print(f"Rasterio still failing: {e}")
else:
    print(f"Server returned error {response.status_code}: {response.text}")