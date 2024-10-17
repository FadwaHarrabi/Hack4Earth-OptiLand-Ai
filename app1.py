import streamlit as st
st.set_page_config(page_title="OptiLand-Ai🌏", layout="centered", initial_sidebar_state="collapsed")
import os
import ee
import torch
import folium
import numpy as np
from streamlit_folium import st_folium
import geopandas as gpd
import warnings
from PIL import Image
from torchvision import models, transforms
import rasterio as rio
from shapely.geometry import box
from tqdm import tqdm
import pandas as pd
from rasterio import mask

import plotly.express as px
import matplotlib.colors as cl

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Earth Engine
ee.Initialize(project="optiland-ai")
temp_tif = "temp.tif"  # Replace with your temp tif path
if os.path.exists(temp_tif):
    os.remove(temp_tif)
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
    'SeaLake': 'blue'  # Changed to 'blue' for better visibility
}

# Load the boundary of the selected province
@st.cache_data
def load_province_options():
    gdf = gpd.read_file("geoboundary/geoboundary.geojson")
    provinces = {row['shapeName']: row for index, row in gdf.iterrows()}
    return provinces

provinces = load_province_options()

# Load a pre-trained model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Sidebar: Province selection
st.sidebar.subheader("Select a governorate")
selected_province = st.sidebar.selectbox("Governorate", ["Tunis", "Monastir", "Mahdia", "Ariana"])
province_shape = provinces[selected_province]['geometry']
province_boundary_gdf = gpd.GeoDataFrame([{'geometry': province_shape}], crs="EPSG:4326")

# Load the pre-loaded TIFF image for each province
@st.cache_data
def load_tif_image(tif_path):
    img = Image.open(tif_path)
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    return img_array.astype(np.uint8)

# Define image paths based on selected province
def get_image_path(province):
    return f"data/images/{province}.tif" if province in ["Tunis", "Monastir", "Ariana", "Mahdia"] else None

# Generate 64x64 polygon tiles over the image
def generate_tiles(image_file, output_file, area_str, size=64):
    raster = rio.open(image_file)
    width, height = raster.shape
    geo_dict = {'id': [], 'geometry': []}
    index = 0

    # Do a sliding window across the raster image
    with tqdm(total=width * height) as pbar:
        for w in range(0, width, size):
            for h in range(0, height, size):
                window = rio.windows.Window(h, w, size, size)
                bbox = rio.windows.bounds(window, raster.transform)
                bbox = box(*bbox)
                uid = f"{area_str.lower().replace(' ', '_')}-{index}"
                geo_dict['id'].append(uid)
                geo_dict['geometry'].append(bbox)
                index += 1
                pbar.update(size * size)

    results = gpd.GeoDataFrame(pd.DataFrame(geo_dict))
    results.crs = {'init': 'epsg:4326'}
    results.to_file(output_file, driver="GeoJSON")
    raster.close()
    return results

# Display satellite image with Folium
def display_satellite_image(image_array, province_shape):
    centroid = province_shape.centroid.coords[0]
    map_obj = folium.Map(location=[centroid[1], centroid[0]], zoom_start=10)

    # Add Google Satellite basemap
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite'
    ).add_to(map_obj)

    # Add a layer for the province boundary
    folium.GeoJson(province_shape, name="Province Boundary").add_to(map_obj)
    folium.LayerControl().add_to(map_obj)
    return map_obj

# Prediction function
def predict_crop(image_path, shape, classes, model):
    with rio.open(image_path) as src:
        out_image, out_transform = rio.mask.mask(src, shape, crop=True)
        _, x_nonzero, y_nonzero = np.nonzero(out_image)
        out_image = out_image[
            :,
            np.min(x_nonzero):np.max(x_nonzero),
            np.min(y_nonzero):np.max(y_nonzero)
        ]

        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        temp_tif = 'temp.tif'
        with rio.open(temp_tif, "w", **out_meta) as dest:
            dest.write(out_image)

        image = Image.open(temp_tif).convert('RGB')
        input_tensor = transform(image)
        output = model(input_tensor.unsqueeze(0))
        _, pred = torch.max(output, 1)
        label = str(classes[int(pred[0])])
        return label

st.title("LULC Prediction Application 🛰️🗺")

# Step 1: Load and display satellite image
image_path = get_image_path(selected_province)
if image_path:
    tif_image = load_tif_image(image_path)
    map_obj = display_satellite_image(tif_image, province_shape)
    st_folium(map_obj, width=800, height=500)

# Step 2: Generate grid tiles
output_file = f"geoboundary/{selected_province}_tiles.geojson"
tiles = generate_tiles(image_path, output_file, selected_province)
image = rio.open(image_path)

# Geopandas sjoin function
tiles = gpd.sjoin(tiles, province_boundary_gdf, predicate='within')

# Define transforms
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Perform predictions
labels = []
for index in tqdm(range(len(tiles)), total=len(tiles)):
    label = predict_crop(image_path, [tiles.iloc[index]['geometry']], classes, model)
    labels.append(label)
tiles['pred'] = labels

# Assign colors to tiles
tiles['color'] = tiles['pred'].apply(lambda x: cl.to_hex(colors.get(x)))

# Create map and add tiles
centroid = province_shape.centroid.coords[0]
map = folium.Map(location=[centroid[1], centroid[0]], zoom_start=10)

# Add Google Satellite basemap
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=True,
    control=True
).add_to(map)

# Add LULC Map with legend
legend_txt = '<span style="color: {col};">{txt}</span>'
for label, color in colors.items():
    name = legend_txt.format(txt=label, col=color)
    feat_group = folium.FeatureGroup(name=name)
    subtiles = tiles[tiles.pred == label]
    if len(subtiles) > 0:
        folium.GeoJson(
            subtiles,
            style_function=lambda feature: {
                'fillColor': feature['properties']['color'],
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.5,
            },
            name='LULC Map'
        ).add_to(feat_group)
        map.add_child(feat_group)

folium.LayerControl().add_to(map)
st_folium(map, width=800, height=500)

# Function to calculate CO2 impact
def calculate_co2_from_tiles(tiles, co2_rates):
    class_counts = tiles['pred'].value_counts()
    co2_total = 0
    for lulc_class, count in class_counts.items():
        co2_rate = co2_rates.get(lulc_class, 0)
        co2_total += co2_rate * count
    return co2_total, class_counts.to_dict()

# Function to display class distribution
def display_class_distribution(class_counts, colors):
    lulc_classes = list(class_counts.keys())
    counts = list(class_counts.values())
    class_colors = [colors[class_name] for class_name in lulc_classes]

    fig = px.bar(
        x=lulc_classes,
        y=counts,
        color=lulc_classes,
        color_discrete_sequence=class_colors,
        labels={'x': 'LULC Class', 'y': 'Number of Tiles'},
        title="Distribution of LULC Classes"
    )

    fig.update_layout(xaxis_title="LULC Class", yaxis_title="Number of Tiles", xaxis_tickangle=-45)
    st.plotly_chart(fig)

# Display the bar plot and CO2 calculation
if st.sidebar.button("Calculate CO₂"):
    co2_total, class_counts = calculate_co2_from_tiles(tiles, co2_rates)
    st.write(f"Total CO₂ impact: {co2_total} tons")
    display_class_distribution(class_counts, colors)
