import numpy as np
import json
from pyproj import Transformer

projection = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=96.00 +lat_0=20.50"
transformer = Transformer.from_crs(
    "epsg:4326",
    projection,
    always_xy=True,
)
transformer_inverse = Transformer.from_crs(
    projection,
    "epsg:4326",
    always_xy=True,
)

coords_segments = [
    [95.96818281, 22.74322665],
    [95.986428, 21.62839242],
    [96.06995685, 20.49512407],
    [96.23976038, 19.27294501],
    [96.47268749, 18.00849782],
]

segments = []
for i in range(len(coords_segments) - 1):
    segments.append((*coords_segments[i], *coords_segments[i + 1]))

# Common properties
common_segment_props = {
    "delay_segment": 0,
    "delta_dip": 5.94,
    "delta_strike": 10.0,
    # "dip": 60.,
    "dip": 90.0,
    "dip_subfaults": 5,
    "hyp_stk": 35,
    "max_vel": 5.5,
    "min_vel": 1.0,
    "neighbours": [],
    "rake": 180.0,
    "rupture_vel": 2.5,
    "stk_subfaults": 39,
}

segment_dicts = []
stk_subfaults_all = []

# Determine northernmost segment for fixed hypocenter
northernmost_idx = np.argmax([(y1 + y2) / 2 for x1, y1, x2, y2 in segments])

for idx, (lon1, lat1, lon2, lat2) in enumerate(segments):
    lons = [lon1, lon2]
    lats = [lat1, lat2]
    x1x2, y1y2 = transformer.transform(lons, lats)

    x1, x2 = x1x2
    y1, y2 = y1y2

    # Compute segment length
    length_m = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    length_km = length_m / 1000.0

    # Compute strike (clockwise from North)
    dx = x2 - x1
    dy = y2 - y1
    strike = np.degrees(np.arctan2(dx, dy)) % 360 + 180

    # delta_strike adjusted so delta_strike * stk_subfaults = length_km
    delta_strike = common_segment_props["delta_strike"]
    stk_subfaults = int(length_km / delta_strike)
    delta_strike = length_km / stk_subfaults

    if y1 < y2:
        # index 0 is top
        lons = lons[::-1]
        lats = lats[::-1]

    hyp_dip = int(
        np.round(
            10
            / (
                np.sin(np.radians(common_segment_props["dip"]))
                * common_segment_props["delta_dip"]
            )
        )
    )

    # Override for northernmost segment
    if idx == northernmost_idx:
        lat = 22.001
        lon = 95.925
        xt, yt = transformer.transform(lon, lat)

        yrel = (yt - y2) / (y1 - y2)
        hyp_stk = int(np.round(yrel * stk_subfaults))

        xi = x2 + yrel * (x1 - x2)
        yi = y2 + yrel * (y1 - y2)
        lat, lon = transformer_inverse.transform(xi, yi)
    else:
        lat = lats[0]
        lon = lons[0]
        hyp_stk = stk_subfaults

    stk_subfaults_all.append(stk_subfaults)

    segment_dict = {
        **common_segment_props,
        "stk_subfaults": stk_subfaults,
        "strike": strike,
        "hyp_dip": hyp_dip,
        "hyp_stk": hyp_stk,
        "delta_strike": delta_strike,
    }

    if idx != northernmost_idx:
        segment_dict |= {"hypocenter": {"lat": lat, "lon": lon, "depth": 10.0}}

    segment_dicts.append(segment_dict)

# Final JSON structure
result = {
    "rise_time": {"delta_rise": 1.5, "min_rise": 1.5, "windows": 5},
    "segments": segment_dicts,
}


# Write JSON to file
output_path = "segments_data.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=4)


# Define slip and rake configuration per segment
segment_slip_configs = []

# Define rake ranges: first segment different from the rest
config = {
    "max_center_slip": 900,
    "max_left_slip": 900,
    "max_lower_slip": 900,
    "max_right_slip": 900,
    "max_slip_delta": 900,
    "max_upper_slip": 900,
    "min_slip": 0,
    "rake_max": 180.0 + 30.0,
    "rake_min": 180.0 - 30.0,
    "rake_step": 21,
    "regularization": {
        "neighbour_down": None,
        "neighbour_left": None,
        "neighbour_right": None,
        "neighbour_up": None,
    },
    "slip_step": 31,
}


for idx, (x1, y1, x2, y2) in enumerate(segments):
    # Create configuration per segment
    config0 = {}
    config0 |= config
    neighbour_left = {"segment": idx, "subfault": 1} if idx > 0 else None
    neighbour_right = (
        {"segment": idx + 2, "subfault": stk_subfaults_all[idx + 1]}
        if idx < len(segments) - 1
        else None
    )

    reg = {
        "neighbour_down": None,
        "neighbour_left": neighbour_left,
        "neighbour_right": neighbour_right,
        "neighbour_up": None,
    }
    config0["regularization"] = reg
    segment_slip_configs.append(config0)

# Write JSON to file
slip_output_path = "model_space.json"
with open(slip_output_path, "w") as f:
    json.dump(segment_slip_configs, f, indent=4)
