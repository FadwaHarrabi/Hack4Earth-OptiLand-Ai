# Satellite-Based Monitoring System for Land Use and Land Cover Changes for Sustainable Development 🌍
![image](https://github.com/user-attachments/assets/e5472acb-2c8c-4dfa-a4dd-71555b2c6a63)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Demo](#demo)
- [Future Enhancements](#future-enhancements)
- [Team](#team)
- [License](#license)

## Introduction
This repository contains an AI-based platform designed to classify **Land Use and Land Cover (LULC)** from **Sentinel-2 satellite imagery**. The platform is developed as part of the [Hack4Earth Hackathon](https://hack4earth.org), focusing on sustainable environmental solutions through AI. It allows users to analyze geospatial data and predict LULC categories such as forests, urban areas, water bodies, and agricultural land, as well as calculate **carbon storage** based on vegetation indices.

## Features
- **Sentinel-2 Image Processing**: Extract multi-band imagery using Google Earth Engine (GEE) for a specific region and time.
- **LULC Classification**: Leverages state-of-the-art deep learning models (e.g., U-Net, ResNet) to classify various land cover types.
- **Carbon Storage Estimation**: Uses vegetation indices (e.g., NDVI) to approximate carbon storage based on biomass density.
- **Interactive Visualization**: Integrates with mapping libraries such as **Folium** and **Leaflet** for interactive map rendering.
- **Sustainable Development Focus**: Aimed at promoting land use practices aligned with environmental sustainability.

## How It Works
The platform operates in three core stages:
1. **Data Acquisition**: Uses GEE API to fetch Sentinel-2 images filtered by cloud coverage and time.
2. **Modeling and Classification**: The LULC classification model is trained on labeled satellite imagery using a CNN-based architecture.
3. **Carbon Storage Calculation**: The NDVI (Normalized Difference Vegetation Index) is calculated from the NIR and Red bands to estimate vegetation biomass and carbon storage.

## Architecture
```mermaid
graph LR
    A[Sentinel-2 Imagery] --> B[Google Earth Engine API]
    B --> C[LULC Model Inference]
    C --> D[NDVI Calculation for Carbon Storage]
    D --> E[Interactive Map Visualization]
    E --> F[User Interaction and Analysis]
