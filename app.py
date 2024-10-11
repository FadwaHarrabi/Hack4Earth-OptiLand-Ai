import streamlit as st
import geemap.foliumap as geemap
import ee
from datetime import datetime

# Initialize the Earth Engine API (Make sure you're authenticated)
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Sidebar input options
st.sidebar.title("LULC Classification for Kairouan, Tunisia")

# Allow users to select specific start and end dates
start_date = st.sidebar.date_input("Start Date", datetime(2021, 1, 1))  # Default to January 1, 2021
end_date = st.sidebar.date_input("End Date", datetime(2021, 12, 31))    # Default to December 31, 2021

# Cloud coverage slider
cloud_coverage = st.sidebar.slider("Max Cloud Coverage (%)", 0, 100, 10)

# Button to trigger the prediction
submit_button = st.sidebar.button("Run LULC Prediction")

# Title
st.title("Land Use and Land Cover (LULC) Prediction for Kairouan, Tunisia")

# Initialize the map centered on Kairouan
Map = geemap.Map(center=[35.6745, 10.1000], zoom=8)

if submit_button:
    try:
        # Define Kairouan region using ADM1 boundary
        tunisia_boundary = ee.FeatureCollection("FAO/GAUL/2015/level1") \
            .filter(ee.Filter.eq('ADM1_NAME', 'Kairouan'))

        # Filter Sentinel-2 image collection for the specified region, date range, and cloud coverage
        image_collection = ee.ImageCollection('COPERNICUS/S2') \
            .filterBounds(tunisia_boundary) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_coverage))

        # Check number of images
        image_count = image_collection.size().getInfo()
        st.write(f"Images found: {image_count}")

        if image_count == 0:
            st.warning(f"No images found for the specified parameters: {start_date} to {end_date}, Cloud Coverage: {cloud_coverage}%")
        else:
            # Create a median composite from the image collection
            image = image_collection.median()

            # Visualization parameters for Sentinel-2 RGB bands
            viz_params = {
                'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue bands
                'min': 0,
                'max': 3000,
                'gamma': 1.4
            }

            # Add the image to the map
            Map.addLayer(image.clip(tunisia_boundary), viz_params, f'Sentinel-2 {start_date} to {end_date}')

            # Add the region boundary
            Map.addLayer(tunisia_boundary, {}, "Kairouan Boundary")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display the map in Streamlit
Map.to_streamlit()
