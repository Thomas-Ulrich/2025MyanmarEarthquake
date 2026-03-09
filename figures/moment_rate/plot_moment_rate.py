# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich, Mathilde Marchandon

import argparse
import glob
import os
import pickle

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def computeMw(label, time, moment_rate):
    M0 = np.trapezoid(moment_rate[:], x=time[:])
    Mw = 2.0 * np.log10(M0) / 3.0 - 6.07
    # print(f"{label} moment magnitude: {Mw} (M0 = {M0:.4e})")
    return Mw


def read_usgs_moment_rate(fname):
    mr_ref = np.loadtxt(fname, skiprows=2)
    # Conversion factor from dyne-cm/sec to Nm/sec (for older usgs files)
    scaling_factor = 1.0 if np.amax(mr_ref[:, 1]) < 1e23 else 1e-7
    mr_ref[:, 1] *= scaling_factor
    return mr_ref


def add_seissol_data(ax, label, fn, plotted_lines, color, linewidth, alpha=1):
    df = pd.read_csv(fn)
    df = df.pivot_table(index="time", columns="variable", values="measurement")
    df["seismic_moment_rate"] = np.gradient(df["seismic_moment"], df.index[1])
    Mw = computeMw(label, df.index.values, df["seismic_moment_rate"])
    if label is not None:
        label = f"{label} (Mw={Mw:.2f})"

    line = ax.plot(
        df.index.values,
        df["seismic_moment_rate"] / scale,
        label=label,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
    )
    plotted_lines.append(line[0])
    return plotted_lines


fig = plt.figure(figsize=(9.5, 3), dpi=80)
ax = fig.add_subplot(111)

ps = 12
scale = 1e19
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)
matplotlib.rcParams["lines.linewidth"] = 0.5
plotted_lines = []


parser = argparse.ArgumentParser(
    description="compare synthetic moment rate releases with observations"
)
parser.add_argument("ensemble_dir", help="path to seissol output file")
parser.add_argument(
    "--best_model", type=str, help='Pattern for best model ((e.g. "dyn_0073")'
)

args = parser.parse_args()


folder_path = os.path.dirname(args.ensemble_dir)
fn = (
    f"{folder_path}/../compiled_results.pkl"
    if folder_path != ""
    else "compiled_results.pkl"
)
with open(fn, "rb") as f:
    df = pickle.load(f)


print(df[["faultfn", "gof_MRF"]].head(10))


lo, hi = np.percentile(df["gof_MRF"].to_numpy(), [5, 95])


cmap = plt.cm.Blues
norm = Normalize(vmin=lo, vmax=1.0)


for _, row in df.sort_values("gof_MRF").iterrows():
    name = row["faultfn"]
    gof = row["gof_MRF"]

    plotted_lines = add_seissol_data(
        ax,
        None,
        f"{args.ensemble_dir}/{name}-energy.csv",
        plotted_lines,
        color=cmap(norm(gof)),
        linewidth=1.2,
        alpha=0.15,
    )


usgs_mr = read_usgs_moment_rate("MomentRateObs/STF_usgs.txt")
Mw = computeMw("usgs", usgs_mr[:, 0], usgs_mr[:, 1])

plotted_lines = []
# Plot Observation
# USGS

line = ax.plot(
    usgs_mr[:, 0],
    usgs_mr[:, 1] / scale,
    label=f"USGS (Mw={Mw:.2f})",
    color="k",
    linestyle=":",
    linewidth=1.2,
)
plotted_lines.append(line[0])

# Scardec
df = pd.read_csv("MomentRateObs/scardec.csv")
Mw = computeMw("Scardec", df["x"], df[" y"])

line = ax.plot(
    df["x"],
    df[" y"] / scale,
    label=f"SCARDEC (Mw={Mw:.2f})",
    color="k",
    linestyle="-.",
    linewidth=1.2,
)
plotted_lines.append(line[0])

# Melgar
df = pd.read_csv("MomentRateObs/Melgar.csv")
Mw = computeMw("Melgar", df["x"], df[" y"])

line = ax.plot(
    df["x"],
    df[" y"] / scale,
    label=f"Melgar et al. (2025) (Mw={Mw:.2f})",
    color="k",
    linestyle="--",
    linewidth=1.2,
)
plotted_lines.append(line[0])

# Plot best model
model_file = glob.glob(f"{args.ensemble_dir}/*{args.best_model}*-energy.csv")

plotted_lines = add_seissol_data(
    ax,
    "dynamic rupture model",
    model_file[0],
    plotted_lines,
    "#0834acff",  # "#083470ff",
    linewidth=1.7,
)

ax.set_ylim(bottom=0)
ax.set_xlim(right=125)
ax.set_xlim(left=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.set_ylabel(r"Moment rate (e19 $\times$ Nm/s)")
ax.set_xlabel("Time (s)")
labels = [lab.get_label() for lab in plotted_lines]
# kargs = {"bbox_to_anchor": (1.0, 1.28)}
kargs = {"bbox_to_anchor": (1.0, 1.1)}
ax.legend(plotted_lines, labels, frameon=False, **kargs)

# plt.legend(frameon=False)
fn = "figures/moment_rate.svg"
plt.savefig(fn)
print(f"done writing {fn}")

# plt.show()
