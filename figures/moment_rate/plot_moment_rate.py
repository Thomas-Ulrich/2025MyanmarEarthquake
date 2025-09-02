# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich, Mathilde Marchandon

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import glob
import os
import argparse

ps = 12
matplotlib.rcParams.update(
    {
        "font.size": ps,  # base font size
        "axes.titlesize": ps,  # title font size
        "axes.labelsize": ps,  # x/y label size
        "xtick.labelsize": ps,
        "ytick.labelsize": ps,
        "font.family": "sans",
        "lines.linewidth": 0.5
    }
)

def computeMw(label, time, moment_rate):
    M0 = np.trapz(moment_rate[:], x=time[:])
    Mw = 2.0 * np.log10(M0) / 3.0 - 6.07
    print(f"{label} moment magnitude: {Mw} (M0 = {M0:.4e})")
    return Mw


def read_usgs_moment_rate(fname):
    mr_ref = np.loadtxt(fname, skiprows=2)
    # Conversion factor from dyne-cm/sec to Nm/sec (for older usgs files)
    scaling_factor = 1.0 if np.amax(mr_ref[:, 1]) < 1e23 else 1e-7
    mr_ref[:, 1] *= scaling_factor
    return mr_ref


fig = plt.figure(figsize=(9.5, 3), dpi=80)
ax = fig.add_subplot(111)

scale = 1e19

plotted_lines = []
os.makedirs("figures", exist_ok=True)


def add_seissol_data(ax, label, fn, plotted_lines, color, linewidth):
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
        linewidth=linewidth,
    )
    plotted_lines.append(line[0])
    return plotted_lines


parser = argparse.ArgumentParser(
    description="compare synthetic moment rate releases with observations"
)
parser.add_argument("ensemble_dir", help="path to seissol output file")

parser.add_argument(
    "--plot_ensemble",
    action="store_true",
    help="plot ensemble as grey lines",
)

parser.add_argument(
    "--best_model", type=str, help='Pattern for best model ((e.g. "dyn_0073")'
)

args = parser.parse_args()

if args.plot_ensemble:
    for fn in glob.glob(f"{args.ensemble_dir}/*energy.csv"):
        label = os.path.basename(fn).replace("-energy.csv", "")
        plotted_lines = add_seissol_data(
            ax, None, fn, plotted_lines, color="#edeeeeff", linewidth=1.2
        )


# Plot best model
model_file = glob.glob(f"{args.ensemble_dir}/*{args.best_model}*-energy.csv")
print(model_file)
assert len(model_file) == 1
plotted_lines = add_seissol_data(
    ax,
    "dynamic rupture model ",
    model_file[0],
    plotted_lines,
    "blue",
    linewidth=1.5,
)


usgs_mr = read_usgs_moment_rate("MomentRateObs/STF_usgs.txt")
Mw = computeMw("usgs", usgs_mr[:, 0], usgs_mr[:, 1])


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
Mw = computeMw("SCARDEC", df["x"], df[" y"])
line = ax.plot(
    df["x"],
    df[" y"] / scale,
    label=f"SCARDEC (Mw={Mw:.2f})",
    color="#f3a966ff",
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
    color="#c448c2ff",
    linestyle="--",
    linewidth=1.2,
)
plotted_lines.append(line[0])

"""
plotted_lines = add_seissol_data(
    ax,
    "triggered asperity",
    #"../seissol_outputs/dyn_0014_coh0.25_0.0_B1.0_C0.2_mud0.25_mus0.6_sn12.0-energy.csv",
    "../seissol_outputs/dyn_0006_coh0.25_0.0_B1.0_C0.1_mud0.2_mus0.6_sn10.0-energy.csv",
    plotted_lines,
    color = "darkgreen",
)
df = pd.read_csv("../data/STF_inoue.csv")
df[" y"] = df[" y"].astype(float)
Mw = computeMw("Inoue et al. (2025)", df["x"], df[" y"])

line = ax.plot(
    df["x"],
    df[" y"] / scale,
    label=f"Inoue et al. (2025) (Mw={Mw:.2f})",
    color="lightgreen",
)
plotted_lines.append(line[0])
"""

ax.set_ylim(bottom=0)
ax.set_xlim(right=150)
ax.set_xlim(left=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.set_ylabel(r"Moment rate (e19 $\times$ Nm/s)")
ax.set_xlabel("Time (s)")
labels = [l.get_label() for l in plotted_lines]
# kargs = {"bbox_to_anchor": (1.0, 1.28)}
kargs = {"bbox_to_anchor": (1.0, 1.1)}
ax.legend(plotted_lines, labels, frameon=False, **kargs)

# plt.legend(frameon=False)
fn = "figures/moment_rate.svg"
plt.savefig(fn, bbox_inches="tight")
print(f"done writing {fn}")

# plt.show()
