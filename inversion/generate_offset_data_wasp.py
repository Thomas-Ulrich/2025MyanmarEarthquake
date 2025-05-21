import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

# Load and simplify the trace
gdf = gpd.read_file("data/surface_rupture_sentinel_v2.shp")
gdf_proj = gdf.to_crs("+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=96.00 +lat_0=20.50")
gdf_simplified_proj = gdf_proj.copy()
gdf_simplified_proj["geometry"] = gdf_proj["geometry"].simplify(1000, preserve_topology=False)
fault_line = gdf_simplified_proj.geometry.union_all()
fault_line_gdf = gpd.GeoDataFrame(geometry=[fault_line], crs=gdf_simplified_proj.crs)
fault_line_latlon = fault_line_gdf.to_crs("EPSG:4326").geometry.values[0]

# Sample regularly along the fault (in projected space!)
fault_length = fault_line.length
spacing = 2000  # meters
num_samples = int(fault_length // spacing)
distances = np.linspace(0, fault_length, num_samples)

# Sample points along fault and compute tangent vectors
sampled_points = [fault_line.interpolate(d) for d in distances]
tangent_vectors = []
for i in range(len(sampled_points)):
    if i == 0:
        p1, p2 = sampled_points[i], sampled_points[i + 1]
    elif i == len(sampled_points) - 1:
        p1, p2 = sampled_points[i - 1], sampled_points[i]
    else:
        p1, p2 = sampled_points[i - 1], sampled_points[i + 1]
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    length = np.hypot(dx, dy)
    tangent_vectors.append((dx / length, dy / length, 0.0))

# Convert sampled points to lat/lon
sampled_points_gdf = gpd.GeoDataFrame(geometry=sampled_points, crs=gdf_proj.crs)
sampled_points_latlon = sampled_points_gdf.to_crs("EPSG:4326")
lons = sampled_points_latlon.geometry.x.values
lats = sampled_points_latlon.geometry.y.values

# Load offset data
offset_df = pd.read_csv("data/Myanmar_sentinel2_offset_latlon_v2.csv")
offset_points = np.vstack((offset_df["lon"], offset_df["lat"])).T
offset_tree = cKDTree(offset_points)

# Interpolate offset at each sampled point using inverse distance weighting
def idw_interpolation(xy, tree, values, radius=0.02, power=2):  # ~2 km radius
    dist, idx = tree.query(xy, k=5, distance_upper_bound=radius)
    mask = np.isfinite(dist)
    if not np.any(mask):
        return np.nan  # no data nearby
    dist = dist[mask]
    idx = idx[mask]
    weights = 1 / dist**power
    return np.sum(weights * values[idx]) / np.sum(weights)

interpolated_offsets = [
    idw_interpolation((lon, lat), offset_tree, offset_df["offset"].values)
    for lon, lat in zip(lons, lats)
]

# Build output DataFrame
output = pd.DataFrame({
    "lon": lons,
    "lat": lats,
    "displacement": interpolated_offsets,
    "sx": [v[0] for v in tangent_vectors],
    "sy": [v[1] for v in tangent_vectors],
    "sz": [0.0] * len(tangent_vectors),
})
output = output.dropna()


# Assuming `output` is a DataFrame with columns: lon, lat, displacement, sx, sy, sz
lons = np.vstack((output["lon"] - 0.01, output["lon"] + 0.01)).T
lat = np.vstack((output["lat"], output["lat"])).T
displacement = np.vstack((-0.5 * output["displacement"], 0.5 * output["displacement"])).T
sx = np.vstack((output["sx"], output["sx"])).T
sy = np.vstack((output["sy"], output["sy"])).T
sz = np.vstack((output["sz"], output["sz"])).T

# Flatten everything row-wise for final DataFrame
final_df = pd.DataFrame({
    "lon": lons.flatten(),
    "lat": lat.flatten(),
    "displacement": displacement.flatten(),
    "sx": sx.flatten(),
    "sy": sy.flatten(),
    "sz": sz.flatten()
})

with open("insar_ascending.txt", "w") as f:
    # Write header with '#'
    f.write("# lon\tlat\tdisplacement\tsx\tsy\tsz\n")
    # Write data
    final_df.to_csv(f, sep="\t", float_format="%.6f", index=False, header=False)
