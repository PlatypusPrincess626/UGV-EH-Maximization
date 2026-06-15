import requests
import rasterio

# 1. Fetch the data once using your bounding box
url = "https://portal.opentopography.org/API/globaldem"
params = {
    'demtype': 'SRTMGL1', # Or other preferred DEM
    'south': 44.4204, 'north': 44.4276,
    'west': -110.5926, 'east': -110.5854,
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