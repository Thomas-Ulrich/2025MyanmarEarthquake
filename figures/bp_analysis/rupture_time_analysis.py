import pandas as pd
from pyproj import Transformer
import numpy as np
import matplotlib.pylab as plt
import trimesh
import seissolxdmf
import argparse
import os


class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def compute_centers(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        return (xyz[connect[:, 0]] + xyz[connect[:, 1]] + xyz[connect[:, 2]]) / 3.0


def get_fault_trace():
    fn = args.fault[0]
    sx = seissolxdmf.seissolxdmf(fn)
    geom = sx.ReadGeometry()
    connect = sx.ReadConnect()
    mesh = trimesh.Trimesh(geom, connect)
    # list vertex of the face boundary
    unique_edges = mesh.edges[
        trimesh.grouping.group_rows(mesh.edges_sorted, require_count=2)
    ]
    unique_edges = unique_edges[:, :, 1]
    ids_external_nodes = np.unique(unique_edges.flatten())

    nodes = mesh.vertices[ids_external_nodes, :]
    nodes = nodes[nodes[:, 2] == 0]
    nodes = nodes[nodes[:, 1].argsort()]

    # Compute strike vector to filter boundaries of near-vertical edges
    grad = np.gradient(nodes, axis=0)
    grad = grad / np.linalg.norm(grad, axis=1)[:, None]

    ids_top_trace = np.where(np.abs(grad[:, 2]) < 0.8)[0]
    nodes = nodes[ids_top_trace]
    return nodes


# parsing python arguments
parser = argparse.ArgumentParser(description="extract slip profile along fault trace")

parser.add_argument("--fault", nargs="+", help="fault xdmf file name", required=True)
parser.add_argument(
    "--downsample", nargs=1, help="take one node every n", default=5, type=int
)
args = parser.parse_args()


plt.rc("font", family="FreeSans", size=8)

myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.92 +lat_0=22.00"
transformer = Transformer.from_crs("epsg:4326", myproj, always_xy=True)
transformeri = Transformer.from_crs(myproj, "epsg:4326", always_xy=True)


trace_nodes = get_fault_trace()[:: args.downsample]
_, lats = transformeri.transform(trace_nodes[:, 0], trace_nodes[:, 1])

# fig = plt.figure(figsize=(5.5, 3.0))
fig, (ax, ax_hist) = plt.subplots(
    1, 2, figsize=(8, 6), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
)

# ax = fig.add_subplot(111)
ax.set_ylabel("latitude")
ax.set_xlabel("rupture time (s)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

lw = 0.8

for i, fn in enumerate(args.fault):
    sx = seissolxdmfExtended(fn)
    fault_centers = sx.compute_centers()
    print(fault_centers.shape)
    _, fault_centers_lat = transformeri.transform(
        fault_centers[:, 0], fault_centers[:, 1]
    )
    idt = sx.ReadNdt() - 1
    RT = np.abs(sx.ReadData("RT", idt))
    ASl = np.abs(sx.ReadData("ASl", idt))

    RT_med = np.zeros_like(lats)
    RTm2sigma = np.zeros_like(lats)
    RTp2sigma = np.zeros_like(lats)
    for ki, lat in enumerate(lats):
        # Determine upper and lower bounds safely
        lat_upper = lats[ki + 1] if ki < len(lats) - 1 else lat - 1.0
        # ki > 0 else lat + 1.0  # or use a realistic max
        lat_lower = lats[ki - 1] if ki > 0 else lat + 1.0
        # ki < len(lats) - 1 else lat - 1.0  # or use a realistic min

        id1 = np.where(
            (fault_centers_lat < lat_upper) & (fault_centers_lat > lat_lower)
        )[0]
        # id1 = np.where(fault_centers_lat[:]>lat)[0]
        # print(lat, len(id1))
        id2 = np.where(RT > 0)[0]
        id1 = np.intersect1d(id1, id2)
        id2 = np.where(ASl > 0.5)[0]
        id1 = np.intersect1d(id1, id2)
        if len(id1) > 0:
            RT_med[ki] = np.median(RT[id1])
            RTm2sigma[ki] = np.percentile(RT[id1], 5)
            RTp2sigma[ki] = np.percentile(RT[id1], 95)
        else:
            RT_med[ki] = np.nan
            RTm2sigma[ki] = np.nan
            RTp2sigma[ki] = np.nan

    ax.plot(
        RT_med,
        lats,
        "royalblue",
        linewidth=lw * (1 + 0.5 * i),
        label="rupture time median",
    )
    ax.plot(
        RTm2sigma,
        lats,
        "royalblue",
        linewidth=lw * (1 + 0.5 * i),
        label="rupture time 95%",
        linestyle=":",
    )
    ax.plot(
        RTp2sigma,
        lats,
        "royalblue",
        linewidth=lw * (1 + 0.5 * i),
        label="rupture time 5%",
        linestyle=":",
    )

for arr in ["au", "eu", "ak"]:
    bp = np.loadtxt(f"filtered_result_{arr}.txt")
    ax.plot(
        bp[:, 2],
        bp[:, 0],
        "x",
        label=f"BP {arr}",
    )

ax.plot(
    50,
    19.78,
    "o",
    label="NPW",
)

ax.legend(loc="lower left")

# Load the TMD aftershock catalog
tmd = pd.read_csv("tmd_catalog_output.txt", delim_whitespace=True)

# Filter lats to 18–23
tmd = tmd[
    (tmd["Latitude"] >= 18)
    & (tmd["Latitude"] <= 23)
    & (tmd["Longitude"] >= 95)
    & (tmd["Longitude"] <= 97)
]
# ["Latitude"]

lat_filtered = tmd["Latitude"]

# Define latitude bins
lat_bins = np.arange(18, 23.2, 0.2)  # step of 0.2
hist_counts, bin_edges = np.histogram(lat_filtered, bins=lat_bins)
lat_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ax_hist.bar(lat_bin_centers, hist_counts, width=0.18, color="gray", alpha=0.6)

# Separate latitudes by magnitude ranges
lat_m1 = tmd[tmd["Magnitude"] < 3.0]["Latitude"]
lat_m2 = tmd[tmd["Magnitude"] >= 3.0]["Latitude"]

# Define latitude bins
lat_bins = np.arange(18, 23.2, 0.2)

# Plot stacked histogram
# fig, ax_hist = plt.subplots(figsize=(8, 3))

ax_hist.hist(
    [lat_m1, lat_m2][::-1],
    bins=lat_bins,
    stacked=True,
    orientation="horizontal",
    color=["lightblue", "green"],
    label=["M < 3", "M ≥ 3"][::-1],
    alpha=0.7,
)
ax_hist.set_xlabel("aftershock count")
# ax_hist.set_ylabel("latitude")
ax_hist.legend()


if not os.path.exists("output"):
    os.makedirs("output")
fn = "output/RT_median_along_strike.pdf"
plt.savefig(fn, dpi=200, bbox_inches="tight")
print(f"done writing {fn}")
plt.show()
