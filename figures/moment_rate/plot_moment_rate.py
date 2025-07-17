import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib


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


fig = plt.figure(figsize=(7.5, 7.5 * 5.0 / 16), dpi=80)
ax = fig.add_subplot(111)

ps = 12
scale = 1e19
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)
matplotlib.rcParams["lines.linewidth"] = 0.5
plotted_lines = []


def add_seissol_data(ax, label, fn, plotted_lines):
    df = pd.read_csv(fn)
    df = df.pivot_table(index="time", columns="variable", values="measurement")
    df["seismic_moment_rate"] = np.gradient(df["seismic_moment"], df.index[1])
    Mw = computeMw(label, df.index.values, df["seismic_moment_rate"])
    line = ax.plot(
        df.index.values,
        df["seismic_moment_rate"] / scale,
        label=f"{label} (Mw={Mw:.2f})",
    )
    plotted_lines.append(line[0])
    return plotted_lines


plotted_lines = add_seissol_data(
    ax,
    "dynamic rupture model",
    "../seissol_outputs/dyn_0080_coh0.25_0.0_B1.1_C0.3_mud0.25_mus0.6_sn10.0-energy.csv",
    plotted_lines,
)

usgs_mr = read_usgs_moment_rate("../data/STF_usgs.txt")
Mw = computeMw("usgs", usgs_mr[:, 0], usgs_mr[:, 1])

line = ax.plot(
    usgs_mr[:, 0],
    usgs_mr[:, 1] / scale,
    label=f"usgs (Mw={Mw:.2f})",
    color="k",
)
plotted_lines.append(line[0])


df = pd.read_csv("../data/scardec.csv")
Mw = computeMw("Scardec", df["x"], df[" y"])
line = ax.plot(
    df["x"], df[" y"] / scale, label=f"Scardec (Mw={Mw:.2f})", color="darkorange"
)
plotted_lines.append(line[0])

df = pd.read_csv("../data/Melgar.csv")
Mw = computeMw("Scardec", df["x"], df[" y"])
line = ax.plot(
    df["x"],
    df[" y"] / scale,
    label=f"Melgar et al. (2025) (Mw={Mw:.2f})",
    color="darkblue",
)
plotted_lines.append(line[0])


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


ax.set_ylim(bottom=0)
ax.set_xlim(right=125)
ax.set_xlim(left=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.set_ylabel(r"Moment rate (e19 $\times$ Nm/s)")
ax.set_xlabel("Time (s)")
labels = [l.get_label() for l in plotted_lines]
kargs = {"bbox_to_anchor": (1.0, 1.28)}
ax.legend(plotted_lines, labels, frameon=False, **kargs)

# plt.legend(frameon=False)
fn = "moment_rate.svg"
plt.savefig(fn)
print(f"done writing {fn}")

plt.show()
