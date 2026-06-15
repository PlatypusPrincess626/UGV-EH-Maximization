import requests

# 1. Fetch the data once using your bounding box
url = "https://portal.opentopography.org/API/globaldem"
params = {
    'demtype': 'SRTMGL1', # Or other preferred DEM
    'south': 44.4204, 'north': 44.4276,
    'west': -110.5926, 'east': -110.5854,
    'outputFormat': 'GTiff',
    'API_Key': 'YOUR_API_KEY'
}
response = requests.get(url, params=params)

# 2. Save and load locally
with open('topo_data.tif', 'wb') as f:
    f.write(response.content)

# Now your simulation loads instantly from 'topo_data.tif'