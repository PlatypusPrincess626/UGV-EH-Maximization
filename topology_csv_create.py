import requests

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

# Check if the request was successful
if response.status_code == 200:
    # Check if the content is actually a TIFF
    if 'image/tiff' in response.headers.get('Content-Type', ''):
        with open('topo_data.tif', 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Error: Expected GeoTIFF but received:")
        print(response.text) # This will print the XML error message
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)