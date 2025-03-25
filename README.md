# PDS_LUTS

# Drone Map Creation and Vehicle Tracking on the Map by Oussama Jaffal

## Overview

Mapping road networks and monitoring vehicle movements is critical for urban planning, traffic management, and the development of autonomous navigation systems. Recent advances in drone technology and computer vision have enabled the capture of detailed aerial imagery and extraction of valuable information. This project utilizes bird’s-eye view drone footage to create a high-resolution map of a scanned area and track vehicle trajectories within it. The integrated system provides a comprehensive visualization of road activity, offering insights into traffic flow and vehicle behavior.

<p align="center">
  <img src="results/final_panorama.jpg" alt="Image 1" width="300" />
  <img src="results/final_panorama_with_trajectories.png" alt="Image 2" width="300" />
</p>


## Workflow

1. **Map Stitching:** The workflow begins with stitching consecutive frames from drone footage to generate a seamless map of the observed region.

   - **How to Use:** Simply execute the Jupyter Notebook `map_stitching.ipynb` to generate the final panorama.
   - **Note:** Some cells in these notebooks are independent and can be run only one time; as a result, certain variables may be re-instantiated during execution.
2. **Vehicle Detection and Tracking:** Vehicle detection and tracking algorithms are applied to identify and monitor vehicles in the footage, with their trajectories overlaid on the generated map.

   - **How to Use:** Execute the Jupyter Notebook `vehicle_tracking.ipynb` to perform vehicle tracking.
     **Note:** Some cells in these notebooks are independent and can be run only one time; as a result, certain variables may be re-instantiated during execution.

## Installation

Before running the notebooks, please install the following repositories:

- **Simple LaMa** (for image inpainting)
  Repository: [Simple LaMa](https://github.com/enesmsahin/simple-lama-inpainting)
- **LightGlue** (for feature matching)
  Repository: [LightGlue](https://github.com/cvg/LightGlue)
- **BoxMOT** (for vehicle tracking)
  Repository: [BoxMOT](https://github.com/mikel-brostrom/boxmot)

Clone or install these projects according to the instructions provided in their respective READMEs.

## Data Organization

- All data should be stored in the existing `Dataset` folder.
- Inside the `Dataset` folder, please include:
  - The `.mov` video file of the drone footage.
  - The `.srt` gps data file.
  - The `detection` folder containing detection data.
- After executing the code, a `tracking` folder will be generated within the `Dataset` directory to store tracking results.

## Report

This report details the methodology, challenges, and results of the project. The proposed approach demonstrates the feasibility and reliability of using drone-based systems for road mapping and traffic analysis, with potential applications in smart city initiatives and transportation research.

## How to Use

- **Map Stitching:**
  Run `map_stitching.ipynb` to generate a high-resolution stitched map of the scanned area.
- **Vehicle Tracking:**
  Run `vehicle_tracking.ipynb` to apply vehicle detection and tracking algorithms.
