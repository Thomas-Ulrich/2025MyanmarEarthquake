import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_insar(df_downsampled, fname):
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_downsampled.lon, df_downsampled.lat)]
    gdf_downsampled = gpd.GeoDataFrame(
        df_downsampled, geometry=geometry, crs="EPSG:4326"
    )

    # Calculate symmetric limits around zero for the norm
    vmin = df_downsampled["displacement"].min()
    vmax = df_downsampled["displacement"].max()
    abs_max = max(abs(vmin), abs(vmax))

    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # Plot using GeoPandas with centered colorbar
    ax = gdf_downsampled.plot(
        column="displacement",
        cmap="seismic",
        markersize=10,
        legend=True,
        figsize=(8, 6),
        norm=norm,  # Apply the centered norm here
    )
    plt.title("Downsampled InSAR Displacement")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    # plt.show()

    plot_vector = True
    if plot_vector:
        # Compute the average vector
        avg_sx = df["sx"].mean()
        avg_sy = df["sy"].mean()
        avg_sz = df["sz"].mean()  # Optional

        # Choose a location to draw the arrow (e.g., mean position)
        center_lon = df["lon"].mean()
        center_lat = df["lat"].mean()

        # Plot the arrow
        ax.quiver(
            center_lon,
            center_lat,  # origin (lon, lat)
            avg_sx,
            avg_sy,  # vector components
            angles="xy",
            scale_units="xy",
            scale=1.0,  # adjust scale for visibility
            color="black",
            width=0.002,
            label="Avg displacement vector",
        )

    fn = os.path.basename(fname)
    plt.savefig(f"{fn}.png")


import glob
import os

folder_path = "data/data_and_model"

shp_files = glob.glob(os.path.join(folder_path, "S*.shp"))
for fname in shp_files:
    print(fname)
    gdf = gpd.read_file(fname)
    # Load your shapefile (update filename if needed)
    # fname = "data/data_and_model/SE1_24MAR2025_05APR2025_D_refined.shp"

    # fname = "data/data_and_model/SE1_19MAR2025_31MAR2025_AZ_OT_D_refined.shp"
    # fname = "data/data_and_model/SE1_22MAR2025_03APR2025_AZ_OT_A_refined.shp"

    gdf = gpd.read_file(fname)

    # Rename and extract necessary columns
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    df = gdf[["lon", "lat", "Observed", "Coef_east", "Coef_north", "Coef_up"]].copy()
    df.columns = ["lon", "lat", "displacement", "sx", "sy", "sz"]

    # Optional: filter out NaNs or bad values
    df = df.dropna(subset=["displacement", "sx", "sy", "sz"])
    plot_insar(df, fname)

    if fname in [
        "data/data_and_model/SE1_19MAR2025_31MAR2025_AZ_OT_D_refined.shp",
        "data/data_and_model/SE1_22MAR2025_03APR2025_AZ_OT_A_refined.shp",
    ]:
        # Save with header starting with #
        letter = fname.split("_")[-2]
        out_fname = "insar_ascending.txt" if letter == "A" else "insar_descending.txt"
        print(len(df))
        # df = df.sample(n=1500, random_state=42)

        with open(out_fname, "w") as f:
            f.write("# lon\tlat\tdisplacement\tsx\tsy\tsz\n")
            df.to_csv(f, sep="\t", float_format="%.6f", index=False, header=False)

        print(f"done writing {out_fname}")

    # plot_insar(df_downsampled)
