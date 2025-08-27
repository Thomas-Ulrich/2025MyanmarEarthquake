#!/usr/bin/env python3
import numpy as np
import datetime
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib
import re
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import cumtrapz


class FaultReceiverData:
    def __init__(self, fname, switchNormal):
        if switchNormal:
            ssign = -1
        else:
            ssign = 1

        with open(fname, "r") as fid:
            fid.readline()
            variablelist = fid.readline().split("=")[1].split(",")
            variablelist = [a.strip().strip('"') for a in variablelist]
            self.x = float(fid.readline().split()[2])
            self.y = float(fid.readline().split()[2])
            self.z = float(fid.readline().split()[2])
            self.Pn0 = float(fid.readline().split()[2])
            self.Ts0 = float(fid.readline().split()[2])
            self.Td0 = float(fid.readline().split()[2])
            # print("location", self.x, self.y, self.z)
            # print("ini stress", self.Pn0, self.Ts0, self.Td0)
            mydata = np.loadtxt(fid)

        self.ndt, ndata = mydata.shape
        self.Time = mydata[:, variablelist.index("Time")]
        self.SRs = mydata[:, variablelist.index("SRs")]
        self.SRd = mydata[:, variablelist.index("SRd")] * ssign

        dt = self.Time[1] - self.Time[0]
        self.Sls = np.cumsum(self.SRs) * dt
        self.Sld = np.cumsum(self.SRd) * dt

        if "T_s" in variablelist:
            self.T_s = mydata[:, variablelist.index("T_s")]
            self.T_s0 = (self.T_s + self.Ts0) * 1e-6
        if "T_d" in variablelist:
            self.T_d = mydata[:, variablelist.index("T_d")]
            self.T_d0 = ssign * (self.T_d + self.Td0) * 1e-6
        if "P_n" in variablelist:
            self.P_n = mydata[:, variablelist.index("P_n")]
            self.P_n0 = (self.P_n + self.Pn0) * 1e-6

        if "StV" in variablelist:
            self.psi = mydata[:, variablelist.index("StV")]
            if np.amax(self.psi) == 0.0:
                self.psi = []
        else:
            self.psi = []

        if "P_f" in variablelist:
            self.P_f = -mydata[:, variablelist.index("P_f")] / 1e6
        else:
            self.P_f = []

        if "Tmp" in variablelist:
            self.Tmp = mydata[:, variablelist.index("Tmp")]
        else:
            self.Tmp = []


def get_rupture_time(SRs):
    threshold_main = 0.15
    threshold_pre = 0.01
    id_pre = None
    # Step 1: First index where SR > 0.15
    indices_main = np.where(SRs > threshold_main)[0]

    if indices_main.size > 0:
        id0 = indices_main[0]

        # Step 2: Last index before id0 where SR > 0.01
        indices_pre = np.where((SRs[:id0] < threshold_pre))[0]
        if indices_pre.size > 0:
            id_pre = indices_pre[-1] + 1
        else:
            id_pre = None  # or handle as needed
            print(f"Note: No SR < {threshold_pre} found before index {id0} in {fname}")
    else:
        print(f"Warning: No SR > {threshold_main} found in {fname}")
    return id_pre


def get_max_cross_correlation(frd, data_Latour):
    dt = 0.01  # sampling interval
    T = frd.Time[-1]
    time_grid = np.arange(0, T, dt)

    # Build interpolated signals of equal length
    interp_model = interp1d(frd.Time, frd.SRs, bounds_error=False, fill_value=0.0)
    model_signal = interp_model(time_grid)

    interp_latour = interp1d(
        data_Latour[:, 0] + 0.5 * T,
        data_Latour[:, 1],
        bounds_error=False,
        fill_value=0.0,
    )
    latour_signal = interp_latour(time_grid)

    # Zero-mean both signals
    model_zm = model_signal - np.mean(model_signal)
    latour_zm = latour_signal - np.mean(latour_signal)

    # Compute cross-correlation
    cross_corr = np.correlate(model_zm, latour_zm, mode="full")
    lags = np.arange(-len(model_zm) + 1, len(model_zm))
    best_lag = lags[np.argmax(cross_corr)]

    # Time shift in seconds
    time_shift = best_lag * dt
    # print(f"Best alignment lag: {best_lag} samples, or {time_shift:.3f} s")
    return 0.5 * T + time_shift


parser = argparse.ArgumentParser(description="compare with Latour slip rate")
parser.add_argument("fault_receiver", help="seissol fault output file")
parser.add_argument(
    "--plot_all",
    action="store_true",
    help="Plot all slip rate functions (default: plot only the 10 best)",
)
parser.add_argument(
    "--align_using_slip_rate_threshold",
    action="store_true",
    help="align signals based slip rate threhold and not cross-correlation",
)
parser.add_argument("--fault_slip", action="store_true", help="plot fault slip")


args = parser.parse_args()
SR_threshold = 0.15

switchNormal = False

rec_files = sorted(glob.glob(f"{args.fault_receiver}*-faultreceiver*"))
rec_files = [fn for fn in rec_files if "dyn-kinmod" not in fn and "fl33" not in fn]

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


fig = plt.figure(figsize=(4.0, 7.5 * 5.0 / 16), dpi=80)
ax = fig.add_subplot(111)

data_Latour = np.loadtxt("Latour.txt", delimiter=",")
# let add some zero after 2s
time_after = np.arange(2, 4, 0.1)
SR_after = np.zeros_like(time_after)
extra_data = np.column_stack((time_after, SR_after))
data_Latour = np.vstack((data_Latour, extra_data))
if not args.align_using_slip_rate_threshold:
    time_before = np.arange(-2, 0, 0.1)
    SR_before = np.zeros_like(time_before)
    extra_data = np.column_stack((time_before, SR_before))
    data_Latour = np.vstack((extra_data, data_Latour))


plt.plot(
    data_Latour[:, 0],
    data_Latour[:, 1],
    label="Latour et al. (2025)",
    marker="o",
    markersize=4,
    linestyle="None",
    color="black",
)

results = {}
wrms = np.full(len(rec_files), np.inf, dtype=float)
results["fault_receiver_fname"] = rec_files
results["slip_rate_rms"] = wrms


for i, fname in enumerate(rec_files):
    frd = FaultReceiverData(fname, switchNormal)

    id0 = get_rupture_time(frd.SRs)
    if not id0:
        continue

    t0 = frd.Time[id0]
    if not args.align_using_slip_rate_threshold:
        t0 = get_max_cross_correlation(frd, data_Latour)

    # Shift time and extract relevant arrays
    time_shifted = frd.Time[id0:] - t0
    sr_shifted = frd.SRs[id0:]

    # Plot preferred model
    if args.plot_all:
        match = re.search(r"dyn[/_-]([^_]+)_", fname)
        if match:
            sim_id = int(match.group(1))
        else:
            sim_id = None  # or raise an error
        plt.plot(time_shifted, sr_shifted, label=f"{sim_id}")

    # Interpolate SR at Latour time points
    interp_func = interp1d(
        time_shifted, sr_shifted, bounds_error=False, fill_value=np.nan
    )
    sr_interp = interp_func(data_Latour[:, 0])

    # Compute RMS error, ignoring NaNs
    valid = ~np.isnan(sr_interp)
    rms = np.sqrt(np.mean((sr_interp[valid] - data_Latour[valid, 1]) ** 2))
    wrms[i] = rms
    print(f"{fname}, onset {t0:.2f}s, RMS error vs Latour: {rms:.4f}")

print(results)

top10_indices = np.argsort(wrms)[:10]

print("10 best models:")
for k, modeli in enumerate(top10_indices[::-1]):
    fname = rec_files[modeli]
    print(10 - k, fname, wrms[modeli])
    if not args.plot_all:
        frd = FaultReceiverData(fname, switchNormal)

        id0 = get_rupture_time(frd.SRs)
        if not id0:
            continue

        t0 = frd.Time[id0]
        if not args.align_using_slip_rate_threshold:
            t0 = get_max_cross_correlation(frd, data_Latour)

        match = re.search(r"dyn[/_-]([^_]+)_", fname)
        if match:
            sim_id = int(match.group(1))
        else:
            sim_id = None  # or raise an error

        # Shift time and extract relevant arrays
        time_shifted = frd.Time[id0:] - t0
        sr_shifted = frd.SRs[id0:]

        # Plot preferred model
        label = f"{sim_id}, {float(wrms[modeli]):.2f}"
        label = "preferred model" if len(top10_indices) == 1 else label

        # Plot preferred model
        (line,) = plt.plot(time_shifted, sr_shifted, label=label)
        if args.fault_slip:
            slip = cumtrapz(sr_shifted, time_shifted, initial=0)
            ax.plot(time_shifted, slip, linestyle="--", color=line.get_color())


dfr = pd.DataFrame(results)
fn = "rms_slip_rate.csv"
dfr.to_csv(fn, index=True, index_label="id")
print(f"done writing {fn}")

if args.align_using_slip_rate_threshold:
    plt.xlim([0, 4])
else:
    plt.xlim([-2, 4])

if len(top10_indices) == 1:
    plt.xlabel(f"Time (s) since {t0:.1f}s post-rupture onset")
    plt.legend(frameon=False, ncol=1, loc="upper right")
else:
    plt.xlabel(f"Time (s) after signal alignment")
    plt.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
    )

# plt.xlabel(f"Time (seconds). time = 0 is {t0:.1f}s after rupture onset in the dynamic rupture model)")
plt.ylabel("Slip rate along strike (m/s)")
if args.fault_slip:
    plt.ylabel("Slip or slip rate along strike (m/s)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()


fn = "CCTV_comparison.svg"
plt.savefig(fn, bbox_inches="tight")
print(f"done writing {fn}")
full_path = os.path.abspath(fn)
print(f"full path: {full_path}")
# plt.show()
