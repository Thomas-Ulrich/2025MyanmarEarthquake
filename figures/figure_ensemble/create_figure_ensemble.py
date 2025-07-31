import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from cmcrameri import cm
import pandas as pd

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


def plot_xy_panel(fig, ax, df, dim_vars, vz, cmap):
    """Generate a 2D plot using variables from 'dim_vars' dict."""

    # Filter by z-value
    df = df[df[dim_vars["z"]["col"]] == vz]

    # Create pivot table for plotting
    pivot = df.pivot_table(
        index=dim_vars["y"]["col"],
        columns=dim_vars["x"]["col"],
        values=dim_vars["v"]["col"],
    )
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    values = pivot.values.astype(float)

    # Plot color mesh
    # im = ax.pcolormesh(X, Y, values, cmap=cmap, shading='auto')
    im = ax.contourf(X, Y, values, cmap=cmap, levels=20)
    if dim_vars["v"]["col"] == "duration":
        contour_lines = ax.contour(
            X, Y, values, levels=[85, 90, 95, 100], colors="k", linestyles="-"
        )
        ax.clabel(contour_lines, inline=True, fontsize=10, fmt="%1.0f")
    ax.set_xlim(pivot.columns.min(), pivot.columns.max())
    ax.set_ylim(pivot.index.min(), pivot.index.max())

    # Overlay hatch for NaNs
    mask_invalid = np.isnan(values)
    ax.pcolor(
        X, Y, np.ma.masked_where(~mask_invalid, mask_invalid), hatch="//", alpha=0
    )

    # Set axis labels and ticks
    ax.set_xticks(pivot.columns.values)
    ax.set_yticks(pivot.index.values)
    if dim_vars["x"]["label"]:
        ax.set_xlabel(dim_vars["x"]["label"])
        ax.set_xticklabels([f"{x:g}" for x in pivot.columns.values])
    else:
        ax.set_xticklabels([])

    if dim_vars["y"]["label"]:
        ax.set_ylabel(dim_vars["y"]["label"])
    else:
        ax.set_yticklabels([])

    # Title and colorbar
    cbar = fig.colorbar(im, ax=ax, label=dim_vars["v"]["label"])


df = pd.read_pickle("compiled_results.pkl")
print(df)
fig, ax = plt.subplots(2, 2, figsize=(12, 8), dpi=80)


dim_vars = {
    "x": {"col": "sigman", "label": r"$\sigma_n$"},
    "y": {"col": "C", "label": "C"},
    "z": {"col": "B", "label": "B"},
}

B = 0.9
for i in range(2):
    for j in range(2):
        dim_vars["x"]["label"] = None if i == 0 else r"$\sigma_n$ (MPa)"
        dim_vars["y"]["label"] = None if j > 0 else "C"
        if (i, j) == (0, 0):
            dim_vars["v"] = {"col": "duration", "label": "Duration (s)"}
        elif (i, j) == (0, 1):
            dim_vars["v"] = {"col": "Mw", "label": "Mw"}
        elif (i, j) == (1, 0):
            dim_vars["v"] = {
                "col": "area_max_R",
                "label": "fault area with R=0.97 (km²)",
            }
        elif (i, j) == (1, 1):
            dim_vars["v"] = {"col": "combined_gof", "label": "combined GOF"}
        plot_xy_panel(fig, ax[i, j], df, dim_vars, vz=B, cmap=cm.cmaps["acton_r"])

        ax[i, j].scatter([13], [0.2], c="g", marker="x")


ax[0, 0].set_title(f"B={B}")

plt.show()
