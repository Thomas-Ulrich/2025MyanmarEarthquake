#!/usr/bin/env python3
import numpy as np
import datetime
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib
import mpl_axes_aligner


ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


class FaultReceiverData:
    def __init__(self, fname, switchNormal=False):
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

        if "u_n" in variablelist:
            self.u_n = mydata[:, variablelist.index("u_n")]

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

class ReceiverData:
    def __init__(self, fname):
        with open(fname, "r") as fid:
            fid.readline()
            variablelist = fid.readline().split("=")[1].split(",")
            variablelist = np.array([a.strip().strip('"') for a in variablelist])
            self.x = float(fid.readline().split()[2])
            self.y = float(fid.readline().split()[2])
            self.z = float(fid.readline().split()[2])
            mydata = np.loadtxt(fid)

        self.ndt, ndata = mydata.shape

        self.Time = mydata[:, variablelist == "Time"].flatten()
        self.u = mydata[:, variablelist == "v1"].flatten()
        self.v = mydata[:, variablelist == "v2"].flatten()
        self.w = mydata[:, variablelist == "v3"].flatten()

rec = ReceiverData('../seissol_outputs/dyn_0080_coh0.25_1.0_B0.95_C0.15_R0.95-receiver-00004-00029.dat')
frec = FaultReceiverData('../seissol_outputs/dyn_0080_coh0.25_1.0_B0.95_C0.15_R0.95-faultreceiver-00001-00029.dat')


# Create two stacked subplots
#fig, (ax1, ax11, ax2) = plt.subplots(nrows=3, sharex=True, figsize=(8, 7), dpi=100)
#fig, (ax1, ax11, ax2) = plt.subplots(nrows=3, sharex=True, figsize=(6.5, 7*2/3), dpi=100)
fig, (ax1, ax11, ax2) = plt.subplots(nrows=3, sharex=True, figsize=(7., 8), dpi=100)

# Bottom panel: linear displacement

ax1.plot(frec.Time, frec.Sls, label="along strike fault slip", color='black')
ax1.set_ylabel("Fault slip (m)")
#ax1.legend()

# Twin y-axis for velocity
ax1b = ax1.twinx()
ax1b.plot(rec.Time, rec.u, label=r"$v_x$", color="tab:blue", linestyle="-")

ax1b.set_ylabel("Velocity (m/s)", color="tab:blue")
ax1b.tick_params(axis='y', labelcolor="tab:blue")
mpl_axes_aligner.align.yaxes(ax1, 0, ax1b, 0, 0.1)

#ax11.set_ylabel("Acceleration (m/s²)")
ax11.set_ylabel("Acceleration (m/s²)")
ax11.plot(rec.Time[:-1], np.diff(rec.u), label=r"$a_x$", color="tab:green", linestyle="-")
ax11.legend(loc="upper left")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")


# Top panel: log-linear acceleration
ax2.semilogy(rec.Time[:-1], np.abs(np.diff(rec.u)), label=r"$a_x$")
ax2.semilogy(rec.Time[:-1], np.abs(np.diff(rec.v)), label=r"$a_y$")
ax2.semilogy(rec.Time[:-1], np.abs(np.diff(rec.w)), label=r"$a_z$")
ax2.axvline(22.7, color='magenta', linestyle='--', linewidth=1, label='P waves')
ax2.axvline(28.5, color='darkblue', linestyle='--', linewidth=1, label='S waves')
ax2.axvline(30.5, color='red', linestyle='--', linewidth=1, label='slip onset')

ax2.set_ylabel("| Acceleration | (m/s²)")
ax2.set_xlabel("Time (s)")
ax2.legend(loc="lower right", ncol=2)


plt.xlim([20, 40])
ax2.set_ylim([1e-8, 1])

# Adjust layout
#plt.tight_layout()
#plt.show()

fn = "CCTV_analysis.svg"
plt.savefig(fn, bbox_inches="tight")
print(f"done writing {fn}")
