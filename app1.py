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
# import os
from tqdm import tqdm
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.express as px
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

# Load a pre-trained model (mockup for demonstration)
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
selected_province = st.sidebar.selectbox("governorate", ["Tunis", "Monastir","Mahdia","Ariana"])

province_shape = provinces[selected_province]['geometry']

# Convert to GeoJSON for Earth Engine
province_geojson = province_shape.__geo_interface__
province_boundary = ee.Geometry(province_geojson)

# Load the pre-loaded TIFF image for each province
@st.cache_data
def load_tif_image(tif_path):
    img = Image.open(tif_path)
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    img_array = img_array.astype(np.uint8)
    return img_array

# Define image paths based on selected province
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

# Generate 64x64 polygon tiles over the image

def generate_tiles(image_file, output_file, area_str, size=64):
    """Generates 64 x 64 polygon tiles.

    Args:
      image_file (str): Image file path (.tif)
      output_file (str): Output file path (.geojson)
      area_str (str): Name of the region
      size(int): Window size

    Returns:
      GeoPandas DataFrame: Contains 64 x 64 polygon tiles
    """
    # Open the raster image using rasterio
    raster = rio.open(image_file)
    width, height = raster.shape

    # Create a dictionary which will contain our 64 x 64 px polygon tiles
    geo_dict = { 'id' : [], 'geometry' : []}
    index = 0

    # Do a sliding window across the raster image
    with tqdm(total=width*height) as pbar:
        for w in range(0, width, size):
            for h in range(0, height, size):
                # Create a Window of your desired size
                window = rio.windows.Window(h, w, size, size)
                # Get the georeferenced window bounds
                bbox = rio.windows.bounds(window, raster.transform)
                # Create a shapely geometry from the bounding box
                bbox = box(*bbox)

                # Create a unique id for each geometry
                uid = '{}-{}'.format(area_str.lower().replace(' ', '_'), index)

                # Update dictionary
                geo_dict['id'].append(uid)
                geo_dict['geometry'].append(bbox)

                index += 1
                pbar.update(size*size)

    # Cast dictionary as a GeoPandas DataFrame
    results = gpd.GeoDataFrame(pd.DataFrame(geo_dict))
    # Set CRS to EPSG:4326
    results.crs = {'init' :'epsg:4326'}
    # Save file as GeoJSON
    results.to_file(output_file, driver="GeoJSON")

    raster.close()
    return results

# Display satellite image with Folium
def display_satellite_image(image_array, province_boundary, grid_tiles=None):
    centroid = province_boundary.centroid().getInfo()['coordinates']
    map_obj = folium.Map(location=[centroid[1], centroid[0]], zoom_start=10)

    # Add Google Satellite basemap
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(map_obj)

    # Overlay the satellite image
    folium.raster_layers.ImageOverlay(
        image=image_array,
        bounds=province_boundary.bounds().getInfo()['coordinates'][0],
        opacity=0.6,
    ).add_to(map_obj)

    # Add a layer for the province boundary
    folium.GeoJson(province_shape, name="Province Boundary").add_to(map_obj)

    # Add grid tiles to the map if they are provided
    if grid_tiles is not None:
        folium.GeoJson(grid_tiles).add_to(map_obj)
    folium.LayerControl().add_to(map_obj)
    return map_obj

# Prediction function (mocked for simplicity)
@st.cache_resource
def predict_lulc(image):
    time.sleep(2)  # Simulate a delay for prediction
    # Simulate LULC prediction (mocking here for demonstration)
    lulc_prediction = np.random.choice(classes, (image.shape[0], image.shape[1]))  # Simulate predictions
    return lulc_prediction





def overlay_lulc_on_map(lulc_prediction, province_boundary):
    # Get the centroid of the province boundary to center the map
    centroid = province_boundary.centroid().getInfo()['coordinates']
    map_obj = folium.Map(location=[centroid[1], centroid[0]], zoom_start=10)

    # Add Google Satellite basemap
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(map_obj)

    # Get unique LULC classes in the prediction
    unique_classes = np.unique(lulc_prediction)

    # Loop over each unique class to create the corresponding layer
    for lulc_class in unique_classes:
        class_str = str(lulc_class)  # Convert the class to string to match the color mapping
        color = colors.get(class_str, 'gray')  # Use gray if the class is not found in the color dictionary

        # Create a mask for the current LULC class
        lulc_mask = lulc_prediction == lulc_class

        if lulc_mask.any():
            # Create GeoJSON-like data for the class
            geojson_data = {
                'type': 'FeatureCollection',
                'features': []
            }

            # Assuming the prediction applies to the entire province boundary
            coords = province_boundary['coordinates'][0]  # Replace with actual feature extraction logic
            geojson_data['features'].append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords],
                },
                'properties': {
                    'fillColor': color,
                    'name': class_str,
                }
            })

            # Create a feature group for the LULC class
            feat_group = folium.FeatureGroup(name=f'{class_str} Distribution')

            # Add the GeoJSON layer with the correct color for the class
            folium.GeoJson(
                geojson_data,
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.5,
                },
                name=f'{class_str} Distribution'
            ).add_to(feat_group)

            # Add the feature group to the map
            map_obj.add_child(feat_group)

    # Add layer control to toggle layers on and off
    folium.LayerControl().add_to(map_obj)

    return map_obj







# def overlay_lulc_on_map(lulc_prediction, province_boundary):
#     centroid = province_boundary.centroid().getInfo()['coordinates']
#     map_obj = folium.Map(location=[centroid[1], centroid[0]], zoom_start=10)

#     # Add Google Satellite basemap
#     folium.TileLayer(
#         tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
#         attr='Google',
#         name='Google Satellite',
#         overlay=True,
#         control=True
#     ).add_to(map_obj)

#     unique_classes = np.unique(lulc_prediction)
#     # st.write(f"Unique LULC classes predicted: {unique_classes}")

#     for label in unique_classes:
#         color = colors.get(label, 'gray')  # Default color
#         lulc_mask = lulc_prediction == label  # Use the label directly
        
#         if lulc_mask.any():
#             coords_list = []
#             # Iterate over mask to extract areas for the given class
#             for i in range(lulc_mask.shape[0]):
#                 for j in range(lulc_mask.shape[1]):
#                     if lulc_mask[i, j]:  # If the pixel belongs to the current class
#                         # Extract the coordinates for the pixel
#                         x, y = j * 64, i * 64  # Example for 64x64 pixels
#                         coords = province_boundary.bounds().getInfo()['coordinates'][0]
#                         coords_list.append(coords)

#             # Create GeoJSON data
#             geojson_data = {
#                 'type': 'FeatureCollection',
#                 'features': []
#             }

#             for coords in coords_list:
#                 geojson_data['features'].append({
#                     'type': 'Feature',
#                     'geometry': {
#                         'type': 'Polygon',
#                         'coordinates': [coords],
#                     },
#                     'properties': {
#                         'fillColor': color,
#                         'name': label,
#                     }
#                 })

#             folium.GeoJson(
#                 geojson_data,
#                 style_function=lambda feature: {
#                     'fillColor': feature['properties']['fillColor'],
#                     'color': 'black',
#                     'weight': 1,
#                     'fillOpacity': 0.5,
#                 },
#                 name=f'{label} Distribution'
#             ).add_to(map_obj)

#     folium.LayerControl().add_to(map_obj)
#     return map_obj


# Main application logic
st.title("LULC Prediction Application🛰️🗺")

# Step 1: Load and display satellite image
image_path = get_image_path(selected_province)
if image_path:
    tif_image = load_tif_image(image_path)
    # st.write("Displaying satellite image...")
    map_obj = display_satellite_image(tif_image, province_boundary)
    st_folium(map_obj, width=800, height=500)

# Step 2: Generate grid tiles
# st.write("Generating grid tiles...")
grid_tiles = generate_tiles(image_path, "geoboundary/grid_tiles.geojson", selected_province)
# st.write("Grid tiles generated successfully!")

# Step 3: Display map with grid tiles
map_with_grid = display_satellite_image(tif_image, province_boundary, grid_tiles=grid_tiles)
# st_folium(map_with_grid, width=700, height=500)

# Step 4: Predict LULC classes (mock)
# st.write("Predicting LULC classes for the image...")
lulc_prediction = predict_lulc(tif_image)

# Step 5: Overlay LULC prediction on the map
# st.write("Overlaying LULC prediction on the map...")
lulc_map = overlay_lulc_on_map(lulc_prediction, province_boundary)
st_folium(lulc_map, width=800, height=500)

def calculate_co2(lulc_prediction, co2_rates):
    """
    Calculate total CO2 emissions or sequestration based on LULC predictions.

    Args:
        lulc_prediction (ndarray): The predicted LULC map.
        co2_rates (dict): A dictionary mapping LULC classes to CO2 emission/sequestration rates.

    Returns:
        float: The total CO2 in tons.
        dict: The count of pixels per LULC class.
    """
    unique_classes, counts = np.unique(lulc_prediction, return_counts=True)
    total_pixels = lulc_prediction.size
    
    co2_total = 0
    lulc_percentage = {}

    for lulc_class, count in zip(unique_classes, counts):
        class_str = str(lulc_class)
        if class_str in co2_rates:
            co2_rate = co2_rates[class_str]
            co2_total += co2_rate * count
        lulc_percentage[class_str] = (count / total_pixels) * 100

    return co2_total, lulc_percentage

def display_lulc_percentage_barplot(lulc_percentage, colors):
    """
    Display a bar plot showing the percentage of LULC classes.

    Args:
        lulc_percentage (dict): A dictionary mapping LULC classes to their percentage.
        colors (dict): A dictionary mapping LULC classes to their corresponding colors.
    """
    lulc_classes = list(lulc_percentage.keys())
    percentages = list(lulc_percentage.values())
    class_colors = [colors[class_name] for class_name in lulc_classes]

    # Create a bar plot using Plotly Express
    fig = px.bar(
        x=lulc_classes,
        y=percentages,
        color=lulc_classes,  # This assigns the colors to the classes
        color_discrete_sequence=class_colors,
        labels={'x': 'LULC Class', 'y': 'Percentage (%)'},
        title="Percentage Distribution of LULC Classes"
    )

    fig.update_layout(xaxis_title="LULC Class", yaxis_title="Percentage (%)", xaxis_tickangle=-45)

    # Display the plot in Streamlit
    st.plotly_chart(fig)



# Display the bar plot

if st.sidebar.button("Calculate CO₂"):
    # Calculate CO2 and display results
    co2_total, lulc_percentage = calculate_co2(lulc_prediction, co2_rates)
    st.write(f"Total CO₂ impact: {co2_total} tons")
    
    # Display the bar plot of LULC class percentages
    display_lulc_percentage_barplot(lulc_percentage,colors)