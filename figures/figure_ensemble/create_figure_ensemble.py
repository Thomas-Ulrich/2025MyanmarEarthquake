import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
import pandas as pd
import copy

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


def plot_xy_panel(fig, ax, df, dim_vars, vz, cmap, contour_lines=None):
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
    if contour_lines:
        contour_lines = ax.contour(
            X, Y, values, levels=contour_lines, colors="k", linestyles="-"
        )
        print(dim_vars["v"]["col"])
        fmt = "%0.1f" if dim_vars["v"]["col"] == "Gc" else "%1.0f"
        ax.clabel(contour_lines, inline=True, fontsize=10, fmt=fmt)
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
        vals = pivot.columns.values
        if len(vals) > 7:
            xticks = [f"{x:g}" if i % 2 == 0 else "" for i, x in enumerate(vals)]
            ax.set_xticklabels(xticks)
        else:
            ax.set_xticklabels([f"{x:g}" for x in vals])
    else:
        ax.set_xticklabels([])

    if dim_vars["y"]["label"]:
        ax.set_ylabel(dim_vars["y"]["label"])
    else:
        ax.set_yticklabels([])

    fig.colorbar(im, ax=ax, label=dim_vars["v"]["label"])


df = pd.read_pickle("compiled_results.pkl")

gofa = pd.read_csv("Gc.csv", sep=",")
gofa["sim_id"] = gofa["faultfn"].str.extract(r"dyn[/_-]([^_]+)_")[0].astype(int)
gofa = gofa[["Gc", "sim_id"]]
df = pd.merge(df, gofa, on="sim_id", how="left")
if "sigman" in df.columns:
    dim_var_x = {"col": "sigman", "label": r"$\sigma_n$"}
elif "R" in df.columns:
    dim_var_x = {"col": "R", "label": "R"}
else:
    raise ValueError("structure of df not understood")

print(df)

for B in df["B"].unique():
    # fig, ax = plt.subplots(2, 2, figsize=(12, 8), dpi=80)
    fig, ax = plt.subplots(2, 2, figsize=(0.7 * 12, 0.7 * 8), dpi=80)

    dim_vars_0 = {
        "x": dim_var_x,
        "y": {"col": "C", "label": "C"},
        "z": {"col": "B", "label": "B"},
    }

    for i in range(2):
        for j in range(2):
            dim_vars = copy.deepcopy(dim_vars_0)
            dim_vars["x"]["label"] = None if i == 0 else dim_vars_0["x"]["label"]
            dim_vars["y"]["label"] = None if j > 0 else dim_vars_0["y"]["label"]
            contour_lines = None
            if (i, j) == (0, 0):
                dim_vars["v"] = {"col": "duration", "label": "duration (s)"}
                contour_lines = [85, 90, 100, 140, 150]
            elif (i, j) == (0, 1):
                dim_vars["v"] = {"col": "Mw", "label": "Mw"}
            elif (i, j) == (1, 0):
                dim_vars["v"] = {
                    "col": "area_max_R",
                    "label": "fault area with R=0.97 (km²)",
                }
                dim_vars["v"] = {
                    "col": "Gc",
                    "label": "fracture energy (MJ/m²)",
                }
                contour_lines = [0.5, 1, 1.5]

            elif (i, j) == (1, 1):
                dim_vars["v"] = {"col": "combined_gof", "label": "combined GOF"}
            plot_xy_panel(
                fig,
                ax[i, j],
                df,
                dim_vars,
                vz=B,
                cmap=cm.cmaps["acton_r"],
                contour_lines=contour_lines,
            )
            if B == 0.95:
                ax[i, j].scatter([0.95], [0.15], c="g", marker="x")

    ax[0, 0].set_title(f"a. B={B}", fontweight="bold")
    ax[0, 1].set_title("b.", fontweight="bold")
    ax[1, 0].set_title("c.", fontweight="bold")
    ax[1, 1].set_title("d.", fontweight="bold")
    ext = "pdf" if B != 0.9 else "svg"
    fn = f"figure_4panelsB{B}.{ext}"
    plt.savefig(fn)
    print(f"done writing {fn}")
