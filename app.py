import streamlit as st
st.set_page_config(page_title="OptiLand-Ai🌏", layout="centered", initial_sidebar_state="collapsed")

import ee
import torch
import folium
import numpy as np
from streamlit_folium import st_folium
import geopandas as gpd
import warnings
from PIL import Image
import time
from torchvision import models
import rasterio as rio
from shapely.geometry import box
import torchvision.transforms as T

from tqdm import tqdm
import pandas as pd
import plotly.express as px
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Earth Engine
ee.Initialize(project="optiland-ai")

# Define LULC classes and colors
co2_rates = {
    'AnnualCrop': 2.5,
    'Forest': -20,
    'HerbaceousVegetation': -5,
    'Highway': 10,
    'Industrial': 50,
    'Pasture': 1.5,
    'PermanentCrop': -3,
    'Residential': 15,
    'River': 0,
    'SeaLake': 0
}

classes = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

colors = {
    'AnnualCrop': 'lightgreen',
    'Forest': 'forestgreen',
    'HerbaceousVegetation': 'yellowgreen',
    'Highway': 'gray',
    'Industrial': 'red',
    'Pasture': 'mediumseagreen',
    'PermanentCrop': 'chartreuse',
    'Residential': 'magenta',
    'River': 'dodgerblue',
    'SeaLake': 'green'
}

# Load the boundary of the selected province
@st.cache_data
def load_province_options():
    gdf = gpd.read_file("geoboundary/geoboundary.geojson")
    provinces = {row['shapeName']: row for index, row in gdf.iterrows()}
    return provinces

# Load a pre-trained model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

model = load_model()
provinces = load_province_options()

# Sidebar: Province selection
st.sidebar.subheader("Select a governorate")
selected_province = st.sidebar.selectbox("governorate", ["Tunis", "Monastir", "Mahdia", "Ariana"])

province_shape = provinces[selected_province]['geometry']

# Convert to GeoJSON for Earth Engine
province_geojson = province_shape.__geo_interface__
province_boundary = ee.Geometry(province_geojson)

# Load the pre-loaded TIFF image for each province
@st.cache_data
def load_tif_image(tif_path):
    with rio.open(tif_path) as raster:
        img_array = raster.read()  # Read the image array
        transform = raster.transform  # Get the transform

    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    img_array = img_array.astype(np.uint8)
    
    return img_array, transform


# Create 'geodata' directory if it doesn't exist
if not os.path.exists("geodata"):
    os.makedirs("geodata")

def get_image_path(province):
    if province == "Tunis":
        return "data/images/Tunis.tif"
    elif province == "Monastir":
        return "data/images/Monastir.tif"
    elif province == "Ariana":
        return "data/images/Ariana.tif"
    elif province == "Mahdia":
        return "data/images/Mahdia.tif"
    else:
        return None

# Generate tiles function
def generate_tiles(image_file, output_file, area_str, size=64):
    raster = rio.open(image_file)
    width, height = raster.shape

    geo_dict = {'id': [], 'geometry': []}
    index = 0

    with tqdm(total=width * height) as pbar:
        for w in range(0, width, size):
            for h in range(0, height, size):
                window = rio.windows.Window(h, w, size, size)
                bbox = rio.windows.bounds(window, raster.transform)
                bbox = box(*bbox)

                uid = '{}-{}'.format(area_str.lower().replace(' ', '_'), index)
                geo_dict['id'].append(uid)
                geo_dict['geometry'].append(bbox)

                index += 1
                pbar.update(size * size)

    results = gpd.GeoDataFrame(pd.DataFrame(geo_dict))
    results.crs = {'init': 'epsg:4326'}
    results.to_file(output_file, driver="GeoJSON")  # Ensure this path is correct

    raster.close()
    return results


def extract_tile(image_array, tile):
    """
    Extracts a tile from the full image array based on the tile's geometry.
    """
    minx, miny, maxx, maxy = tile['geometry'].bounds
    
    # Convert the bounds to pixel indices
    height, width = image_array.shape[0:2]
    
    pixel_minx = int((minx - tile.transform[2]) / tile.transform[0])
    pixel_miny = int((maxy - tile.transform[5]) / tile.transform[4])
    pixel_maxx = int((maxx - tile.transform[2]) / tile.transform[0])
    pixel_maxy = int((miny - tile.transform[5]) / tile.transform[4])

    pixel_minx = max(0, pixel_minx)
    pixel_miny = max(0, pixel_miny)
    pixel_maxx = min(width, pixel_maxx)
    pixel_maxy = min(height, pixel_maxy)

    # Extract the tile
    tile_image = image_array[pixel_miny:pixel_maxy, pixel_minx:pixel_maxx]
    
    return tile_image

def predict_lulc_for_tiles(grid_tiles, image_array, raster_transform):
    predictions = []
    
    for _, tile in grid_tiles.iterrows():
        # Extract the tile from the full image
        tile_image = extract_tile(image_array, tile)
        
        # Preprocess the tile image
        input_tensor = preprocess_image(tile_image)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        _, predicted_class = torch.max(outputs, dim=1)
        predicted_class = predicted_class.squeeze(0).cpu().numpy()
        
        predictions.append(predicted_class)

    return np.array(predictions)

# Display satellite image with Folium
def display_satellite_image(image_array, province_boundary, grid_tiles=None):
    centroid = province_boundary.centroid().getInfo()['coordinates']
    map_obj = folium.Map(location=[centroid[1], centroid[0]], zoom_start=10)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(map_obj)

    folium.raster_layers.ImageOverlay(
        image=image_array,
        bounds=province_boundary.bounds().getInfo()['coordinates'][0],
        opacity=0.6,
    ).add_to(map_obj)

    folium.GeoJson(province_shape, name="Province Boundary").add_to(map_obj)

    if grid_tiles is not None:
        folium.GeoJson(grid_tiles).add_to(map_obj)
    folium.LayerControl().add_to(map_obj)
    return map_obj

# Prediction function
def preprocess_image(image):
    pil_img = Image.fromarray(image)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(pil_img).unsqueeze(0)
    return img_tensor

@st.cache_resource
def predict_lulc(image_array):
    input_tensor = preprocess_image(image_array)
    with torch.no_grad():
        outputs = model(input_tensor)
    
    _, predicted_classes = torch.max(outputs, dim=1)
    predicted_classes = predicted_classes.squeeze(0).cpu().numpy()
    
    class_labels = np.vectorize(lambda x: classes[x])(predicted_classes)
    
    return class_labels

def overlay_lulc_on_map(lulc_prediction, province_boundary, grid_tiles):
    centroid = province_boundary.centroid().getInfo()['coordinates']
    map_obj = folium.Map(location=[centroid[1], centroid[0]], zoom_start=10)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(map_obj)

    for i, grid_tile in grid_tiles.iterrows():
        if i < len(lulc_prediction):
            lulc_class = lulc_prediction[i]
            color = colors[lulc_class]
            folium.GeoJson(grid_tile['geometry'], style_function=lambda x, color=color: {
                'fillColor': color,
                'color': color,
                'weight': 1,
                'fillOpacity': 0.5
            }).add_to(map_obj)

    folium.GeoJson(province_shape, name="Province Boundary").add_to(map_obj)
    folium.LayerControl().add_to(map_obj)

    return map_obj

# Main application
def main():
    st.title("OptiLand-Ai 🌏")
    st.sidebar.title("LULC Classification")

    # Image loading and processing
    image_path = get_image_path(selected_province)
    if image_path:
        image_array, transform = load_tif_image(image_path)

        # Generate tiles
        tile_geojson_path = f"geodata/{selected_province.lower().replace(' ', '_')}_tiles.geojson"
        grid_tiles = generate_tiles(image_path, tile_geojson_path, selected_province)

        # Prediction
        lulc_prediction = predict_lulc_for_tiles(grid_tiles, image_array, transform)
        
        # Displaying results
        map_obj = overlay_lulc_on_map(lulc_prediction, province_boundary, grid_tiles)
        st_folium(map_obj, width=700, height=500)

if __name__ == "__main__":
    main()
