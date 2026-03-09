#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich, Mathilde Marchandon

import pandas as pd
from pyproj import Transformer
import numpy as np
import matplotlib.pylab as plt
import trimesh
import seissolxdmf
from scipy import spatial
import os
import re
import argparse
from matplotlib.colors import Normalize
from tqdm import tqdm
import pickle


plt.rc("font", size=12)


class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def compute_centers(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        centers = np.sum(xyz[connect], axis=1) / 3.0
        return centers

    def compute_strike(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        strike = np.zeros((connect.shape[0], 2))

        for i in range(connect.shape[0]):
            p0 = xyz[connect[i, 0]]
            p1 = xyz[connect[i, 1]]
            p2 = xyz[connect[i, 2]]

            # Vectors in the triangle plane
            v1 = p1 - p0
            v2 = p2 - p0

            # Normal vector to the triangle (fault) plane
            normal = np.cross(v1, v2)

            # Horizontal projection of the normal vector (zero z-component)
            horizontal_normal = np.array([normal[0], normal[1], 0.0])
            if np.linalg.norm(horizontal_normal) < 1e-8:
                raise ValueError("fault is flat (normal has no horizontal component")

            # Strike is perpendicular to horizontal normal, lying in XY plane
            strike_vec = np.cross([0, 0, 1], horizontal_normal)
            strike_vec = strike_vec[:2] / np.linalg.norm(strike_vec[:2])

            strike[i, :] = strike_vec

        return strike

    def get_fault_trace(self, threshold_z):
        geom = self.ReadGeometry()
        connect = self.ReadConnect()

        mesh = trimesh.Trimesh(vertices=geom, faces=connect)

        # Boundary edges: edges that appear exactly once (i.e., on the boundary)
        boundary_edges = mesh.edges[
            trimesh.grouping.group_rows(mesh.edges_sorted, require_count=2)
        ][:, :, 1]
        ids_external_nodes = np.unique(boundary_edges)

        # Extract boundary nodes at surface (z=0), sorted by y
        nodes = mesh.vertices[ids_external_nodes]
        nodes = nodes[nodes[:, 2] >= threshold_z]
        nodes = nodes[np.argsort(nodes[:, 1])]

        # Filter out steep (near-vertical) segments using z-gradient
        grad = np.gradient(nodes, axis=0)
        grad /= np.linalg.norm(grad, axis=1)[:, None]
        nodes = nodes[np.abs(grad[:, 2]) < 0.8]

        return nodes


def extract_dyn_number(filename):
    match = re.search(r"dyn_(\d+)", filename)
    return int(match.group(1)) if match else -1


def plot_individual_offset_figure(df, acc_dist, slip_at_trace, fname):
    fig = plt.figure(figsize=(7.5, 3.0))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Fault offsets (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    lw = 0.8
    ax.errorbar(
        acc_dist,
        df["offset"],
        yerr=df["error"],
        color="royalblue",
        linestyle="-",
        linewidth=lw / 2.0,
        label="Inferred offset",
        marker="o",
        markersize=2,
    )

    ax.plot(
        acc_dist,
        slip_at_trace,
        "royalblue",
        linewidth=1,
        label="Predicted offset",
    )

    ax.text(-79, 6.5, "North", fontweight="medium")
    ax.text(431, 6.5, "South", fontweight="medium")

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"done writing {fname}")
    plt.close(fig)


def init_all_offsets_figure(acc_dist, dfo):
    "init plot with every model"

    fig = plt.figure(figsize=(9.5, 3.0))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Fault offsets (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    lw = 0.8
    ax.errorbar(
        acc_dist,
        dfo["offset"],
        yerr=dfo["error"],
        color="k",
        linestyle="-",
        linewidth=lw / 2.0,
        label="Inferred offset",
        marker="o",
        markersize=2,
        zorder=2,
    )

    ax.text(-79, 6.5, "North", fontweight="medium")
    ax.text(400, 6.5, "South", fontweight="medium")
    return fig, ax


def remove_spikes(data, threshold=5.0):
    """
    Remove spikes from a 1D numpy array using gradient thresholding.
    Parameters:
        data (np.ndarray): Input 1D array.
        threshold (float): Threshold on gradient to detect spikes.
    Returns:
        np.ndarray: Array with spikes replaced by interpolated values.
    """
    data = data.copy()
    grad = np.gradient(data)

    # Identify spike locations where gradient is abnormally large
    spike_indices = np.where(np.abs(grad) > threshold)[0]
    for idx in spike_indices:
        if idx + 2 in spike_indices:
            if 0 <= idx < len(data) - 2:
                # Replace spike with average of neighbors
                data[idx + 1] = 0.5 * (data[idx] + data[idx + 2])

    return data


def compute_rms_offset(folder, offset_data, bestmodel, threshold_z):
    # Read optical offset
    dfo = pd.read_csv(offset_data, sep=",")
    dfo = dfo.sort_values(by="lat", ascending=False)
    print(dfo)

    # Transform lat lon in DR models projection and compute distance
    myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.93 +lat_0=22.00"
    transformer = Transformer.from_crs("epsg:4326", myproj, always_xy=True)
    x, y = transformer.transform(dfo["lon"].to_numpy(), dfo["lat"].to_numpy())
    xy = np.vstack((x, y)).T
    dist = np.linalg.norm(xy[1:, :] - xy[0:-1, :], axis=1)
    acc_dist = np.add.accumulate(dist) / 1e3
    acc_dist = np.insert(acc_dist, 0, 0) - 69

    # Create figure folder
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # List all models in DR output folder
    folder_path = os.path.dirname(folder)

    with open(f"{folder_path}/../compiled_results.pkl", "rb") as f:
        df = pickle.load(f)

    # 1) Trier par gof_offsets (du pire au meilleur, par ex.)
    df_sorted = df.sort_values("gof_offsets", ascending=True)

    mask_main = df_sorted["faultfn"].str.contains("dyn_0080", regex=False)

    # Separate bestmodel from the others...
    df_main = df_sorted[mask_main]  # best model
    df_others = df_sorted[~mask_main]  # tous les autres

    # ... and put it at the end of the list (for plotting reasons)
    df_final = pd.concat([df_others, df_main], ignore_index=True)

    lo, hi = np.percentile(df["gof_offsets"].to_numpy(), [5, 95])

    cmap = plt.cm.Blues
    norm = Normalize(vmin=lo, vmax=1.0)

    print(df_final)

    #################################################################
    # Compute weighted RMS and plot individual figure of each model #
    #################################################################

    fig, ax = init_all_offsets_figure(acc_dist, dfo)
    # Loop on each model in the output filder
    for _, row in tqdm(df_final.iterrows(), total=df_final.shape[0]):
        base_name = row["faultfn"]
        gof = row["gof_offsets"]
        fault = f"{folder_path}/{base_name}_compacted-fault.xdmf"

        if bestmodel in base_name:
            color = "#0834acff"
            alpha = 1
            linewidth = 1.2
        else:
            color = cmap(norm(gof))  # "#edeeeeff"
            alpha = 0.2
            linewidth = 1.2

        # Read SeisSol output
        sx = seissolxdmfExtended(f"{fault}")
        fault_centers = sx.compute_centers()
        idt = sx.ReadNdt() - 1
        Sls = np.abs(sx.ReadData("Sls", idt))

        # Find indices of surface subfaults
        trace_nodes = sx.get_fault_trace(threshold_z)[::1]
        tree = spatial.KDTree(fault_centers)
        dist, idsf = tree.query(trace_nodes)
        # Surface fault slip
        slip_at_trace = Sls[idsf]

        # Surface fault slip at observation location
        tree = spatial.KDTree(trace_nodes[:, 0:2])
        dist, idsf2 = tree.query(xy)
        slip_at_trace = slip_at_trace[idsf2]

        slip_at_trace = remove_spikes(slip_at_trace, threshold=1.0)

        ax.plot(
            acc_dist,
            slip_at_trace,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label="Predicted offset",
        )

    ax.errorbar(
        acc_dist,
        dfo["offset"],
        yerr=dfo["error"],
        color="k",
        linestyle="-",
        linewidth=0.8 / 2.0,
        label="Inferred offset",
        marker="o",
        markersize=2,
        zorder=2,
    )

    ax.plot(
        [121.94, 246.283],
        [0, 0],
        color="k",
        marker="|",
        linestyle="None",
    )
    ax.text(
        121.94,
        0.2,
        "CCTV",
        color="k",
        fontweight="medium",
        horizontalalignment="center",
    )
    ax.text(
        246.283,
        0.2,
        "NPW",
        color="k",
        fontweight="medium",
        horizontalalignment="center",
    )

    fn = "figures/comparison_offset_all_models.pdf"
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"done writing {fn}")
    fn = "figures/comparison_offset_all_models.svg"
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"done writing {fn}")


parser = argparse.ArgumentParser(
    description="""compute fit (RMS) to offset of models from an
    ensemble of DR models."""
)

parser.add_argument("output_folder", help="folder where the models lie")
parser.add_argument("offset_data", help="path to offset data")
parser.add_argument("bestmodel", help='Pattern for best model (e.g. "dyn_0080")')
parser.add_argument(
    "--threshold_z",
    help="threshold depth used for selecting fault trace nodes",
    type=float,
    default=0,
)

args = parser.parse_args()
compute_rms_offset(
    args.output_folder,
    args.offset_data,
    args.bestmodel,
    args.threshold_z,
)
