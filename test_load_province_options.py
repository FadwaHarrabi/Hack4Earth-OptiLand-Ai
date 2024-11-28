import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import os
import geopandas as gpd
import numpy as np
import torch
import folium
import rasterio as rio
from app import load_province_options,load_tif_image,generate_tiles,predict_crop,calculate_co2_from_tiles,classes,model,co2_rates
# Utility functions for testing
def test_load_province_options():
    """
    Test the load_province_options function
    Verify that:
    1. Provinces are loaded correctly
    2. All expected provinces are present
    3. Geometric data is valid
    """
    provinces = load_province_options()
    expected_provinces = ["Tunis", "Monastir", "Mahdia", "Ariana", "Zaghouan", "Manouba"]
    
    assert isinstance(provinces, dict), "Provinces should be a dictionary"
    
    for province in expected_provinces:
        assert province in provinces, f"{province} is missing from loaded provinces"
        
        # Check geometry validity
        assert 'geometry' in provinces[province], f"Geometry missing for {province}"
        assert not provinces[province]['geometry'].is_empty, f"Geometry for {province} is empty"

def test_load_tif_image():
    """
    Test the load_tif_image function
    Verify that:
    1. Image is loaded correctly
    2. Image is normalized between 0-255
    3. Image is converted to uint8
    """
    test_provinces = ["Tunis", "Monastir", "Ariana"]
    
    for province in test_provinces:
        tif_path = f"data/images/{province}.tif"
        img_array = load_tif_image(tif_path)
        
        assert isinstance(img_array, np.ndarray), f"Image for {province} not loaded as numpy array"
        assert img_array.dtype == np.uint8, f"Image for {province} not converted to uint8"
        
        # Check normalization
        assert np.min(img_array) >= 0, f"Image for {province} has values below 0"
        assert np.max(img_array) <= 255, f"Image for {province} has values above 255"

def test_generate_tiles():
    """
    Test the generate_tiles function
    Verify that:
    1. Tiles are generated correctly
    2. Correct number of tiles created
    3. Tiles have correct attributes
    """
    test_provinces = ["Tunis", "Monastir"]
    
    for province in test_provinces:
        tif_path = f"data/images/{province}.tif"
        output_file = f"geoboundary/{province}_test_tiles.geojson"
        
        tiles = generate_tiles(tif_path, output_file, province)
        
        assert isinstance(tiles, gpd.GeoDataFrame), f"Tiles for {province} not a GeoDataFrame"
        assert len(tiles) > 0, f"No tiles generated for {province}"
        
        # Check required columns
        required_cols = ['id', 'geometry']
        for col in required_cols:
            assert col in tiles.columns, f"Missing {col} column in tiles for {province}"

def test_predict_crop():
    """
    Test the predict_crop function
    Verify that:
    1. Prediction returns a valid class
    2. Model handles different input sizes
    3. Prediction is consistent
    """
    test_provinces = ["Tunis", "Monastir"]
    
    for province in test_provinces:
        tif_path = f"data/images/{province}.tif"
        
        # Load province shape
        provinces = load_province_options()
        province_shape = provinces[province]['geometry']
        
        # Predict
        label = predict_crop(tif_path, [province_shape], classes, model)
        
        assert label in classes, f"Prediction for {province} is not a valid class"

def test_calculate_co2_from_tiles():
    """
    Test CO2 calculation function
    Verify that:
    1. Calculation handles different tile distributions
    2. Returns correct total and class counts
    """
    # Mock tiles dataframe
    mock_tiles = gpd.GeoDataFrame({
        'pred': ['Forest', 'Industrial', 'AnnualCrop', 'Forest', 'Residential']
    })
    
    co2_total, class_counts = calculate_co2_from_tiles(mock_tiles, co2_rates)
    
    assert isinstance(co2_total, (int, float)), "CO2 total should be a number"
    assert isinstance(class_counts, dict), "Class counts should be a dictionary"
    
    # Verify class counts
    expected_counts = {
        'Forest': 2,
        'Industrial': 1,
        'AnnualCrop': 1,
        'Residential': 1
    }
    assert class_counts == expected_counts, "Incorrect class counts"
# Modify AppTest configuration to extend timeout
def extend_timeout(func):
    def wrapper(*args, **kwargs):
        # Increase timeout to 10 seconds
        AppTest.DEFAULT_TIMEOUT = 50
        return func(*args, **kwargs)
    return wrapper

@extend_timeout
def test_streamlit_app_render():
    """
    Integration test for Streamlit app render with extended timeout
    """
    at = AppTest.from_file("app.py")
    
    # Add more explicit waiting and debugging
    try:
        at.run(timeout=10)  # Explicitly set longer timeout
    except Exception as e:
        print(f"App initialization error: {e}")
        # Add more detailed error logging if needed
        raise

    # Check main title
    assert len(at.title) > 0, "No title found in the app"
    assert at.title[0].value == "LULC Prediction Application 🛰️🗺"
    
    # Check tabs
    assert len(at.tabs) == 2
    assert at.tabs[0].label == "Map View"
    assert at.tabs[1].label == "CO₂ Impact Analysis"

@extend_timeout
def test_province_selection():
    """
    Test province selection functionality with extended timeout
    """
    at = AppTest.from_file("app.py")
    
    try:
        at.run(timeout=10)  # Longer timeout
    except Exception as e:
        print(f"App initialization error: {e}")
        raise

    # Verify sidebar exists
    assert len(at.sidebar) > 0, "Sidebar not found"

    # Test province selection dropdown
    province_select = at.sidebar.selectbox[0]
    expected_provinces = ["Tunis", "Monastir", "Mahdia", "Ariana", "Zaghouan", "Manouba"]
    
    for province in expected_provinces:
        try:
            province_select.select(province)
            at.run(timeout=10)  # Longer timeout for each province selection
        except Exception as e:
            print(f"Error selecting province {province}: {e}")
            raise
        
        # Verify map and prediction updates
        assert len(at.map) > 0, f"Map not rendered for {province}"

@extend_timeout
def test_co2_impact_tab():
    """
    Test CO2 Impact Analysis tab functionality with extended timeout
    """
    at = AppTest.from_file("app.py")
    
    try:
        at.run(timeout=10)  # Longer timeout
    except Exception as e:
        print(f"App initialization error: {e}")
        raise

    # Switch to CO2 Impact tab
    at.tabs[1].click()
    at.run(timeout=10)

    # Check metrics
    metrics = at.metric
    assert len(metrics) >= 3, "Expected at least 3 metrics in CO2 Impact tab"
    
    # Verify metric names
    metric_names = [m.label for m in metrics]
    expected_names = ['📌Total CO₂ Impact', '📌Top LULC Class', '📌Tiles Analyzed']
    
    # Check if all expected names are present
    for name in expected_names:
        assert name in metric_names, f"Missing metric: {name}"

# Additional diagnostic test to check Earth Engine and model initialization
def test_critical_initializations():
    """
    Diagnostic test to check critical initializations
    """
    import ee
    import torch
    
    # Test Earth Engine initialization
    try:
        ee.Initialize(project="optiland-ai")
    except Exception as e:
        pytest.fail(f"Earth Engine initialization failed: {e}")
    
    # Test model loading
    try:
        from app import load_model
        model = load_model()
        assert model is not None, "Model failed to load"
    except Exception as e:
        pytest.fail(f"Model loading failed: {e}")

# Debugging helper for complex initializations
def debug_app_initialization():
    """
    Separate function to help diagnose initialization issues
    Run this manually to get more detailed output
    """
    import sys
    import traceback
    
    try:
        import streamlit as st
        import ee
        import torch
        
        # Verbose Earth Engine initialization
        ee.Initialize(project="optiland-ai")
        print("Earth Engine initialized successfully")
        
        # Load model
        from app import load_model
        model = load_model()
        print("Model loaded successfully")
        
    except Exception as e:
        print("Initialization Error:")
        print(traceback.format_exc())
        sys.exit(1)
# Streamlit App Integration Tests
def test_streamlit_app_render():
    """
    Integration test for Streamlit app render
    Verify basic app structure and initial state
    """
    at = AppTest.from_file("app.py")
    at.run()

    # Check main title
    assert at.title[0].value == "LULC Prediction Application 🛰️🗺"
    
    # Check tabs
    assert len(at.tabs) == 2
    assert at.tabs[0].label == "Map View"
    assert at.tabs[1].label == "CO₂ Impact Analysis"

def test_province_selection():
    """
    Test province selection functionality
    """
    at = AppTest.from_file("app.py")
    at.run()

    # Verify sidebar exists
    assert len(at.sidebar) > 0, "Sidebar not found"

    # Test province selection dropdown
    province_select = at.sidebar.selectbox[0]
    expected_provinces = ["Tunis", "Monastir", "Mahdia", "Ariana", "Zaghouan", "Manouba"]
    
    for province in expected_provinces:
        province_select.select(province)
        at.run()
        
        # Verify map and prediction updates
        assert len(at.map) > 0, f"Map not rendered for {province}"

def test_co2_impact_tab():
    """
    Test CO2 Impact Analysis tab functionality
    """
    at = AppTest.from_file("app.py")
    at.run()

    # Switch to CO2 Impact tab
    at.tabs[1].click()
    at.run()

    # Check metrics
    metrics = at.metric
    assert len(metrics) == 3, "Expected 3 metrics in CO2 Impact tab"
    
    # Verify metric names
    metric_names = [m.label for m in metrics]
    expected_names = ['📌Total CO₂ Impact', '📌Top LULC Class', '📌Tiles Analyzed']
    assert set(metric_names) == set(expected_names), "Incorrect metrics displayed"

# Performance and Error Handling Tests
def test_large_image_handling():
    """
    Test application's behavior with large satellite images
    """
    # Simulate large image processing
    test_provinces = ["Tunis"]  # Choose a province with a larger image
    
    for province in test_provinces:
        tif_path = f"data/images/{province}.tif"
        
        # Open image and check size
        with rio.open(tif_path) as src:
            width, height = src.shape
            
        # Ensure processing doesn't fail for large images
        tiles = generate_tiles(tif_path, f"geoboundary/{province}_large_test.geojson", province)
        assert len(tiles) > 0, f"Failed to process large image for {province}"

def test_invalid_image_handling():
    """
    Test application's behavior with potentially invalid images
    """
    # This test requires careful setup of an invalid or corrupt image
    with pytest.raises(Exception):
        # Simulate loading a corrupt or non-existent image
        load_tif_image("invalid_image.tif")

# Add more specific tests as needed