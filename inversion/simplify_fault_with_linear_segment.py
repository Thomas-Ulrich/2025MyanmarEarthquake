#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 based on an initial script of Mathilde Marchandon
"""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

gdf = gpd.read_file("data/surface_rupture_sentinel_v2.shp")
# Step 2: Project to UTM or any metric CRS for accurate distance-based simplification
# You may want to select an appropriate UTM zone manually
# gdf_proj = gdf.to_crs("EPSG:32646")  # Example: UTM zone 46N (covers part of Myanmar)
gdf_proj = gdf.to_crs("+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=96.00 +lat_0=20.50")  # Example: UTM zone 45N

# Step 3: Apply Douglas-Peucker simplification (epsilon in meters)
epsilon = 1000  # meters
gdf_simplified_proj = gdf_proj.copy()
gdf_simplified_proj["geometry"] = gdf_proj["geometry"].simplify(
    tolerance=epsilon, preserve_topology=False
)

# Step 4: Convert simplified geometries back to lat/lon
gdf_simplified = gdf_simplified_proj.to_crs("EPSG:4326")

for geom in gdf_simplified.geometry:
    x, y = geom.xy
    coords = np.vstack((x, y)).T
    print(coords)

# Step 5: Plot original and simplified lines in lat/lon
fig, ax = plt.subplots(figsize=(10, 6))

gdf_latlon = gdf.to_crs("EPSG:4326")

gdf_latlon.plot(ax=ax, color="blue", linewidth=1, label="Original Line")

# Plot simplified points as "x" markers
for geom in gdf_simplified.geometry:
    x, y = geom.xy
    ax.plot(x, y, "x", color="red", label="Simplified Points")

ax.legend()
ax.set_title(f"Douglas-Peucker Simplification (ε = {epsilon} m)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
