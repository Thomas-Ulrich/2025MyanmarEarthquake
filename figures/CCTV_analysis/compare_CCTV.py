#!/usr/bin/env python3
import numpy as np
import datetime
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib


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
            print("location", self.x, self.y, self.z)
            print("ini stress", self.Pn0, self.Ts0, self.Td0)
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


fname = "/home/ulrich/trash/dyn_0017_coh0.25_0.0_B1.0_C0.2_mud0.3_mus0.6_sn15.0-faultreceiver-00001-00006.dat"
switchNormal = False
frd = FaultReceiverData(fname, switchNormal)

id0 = np.where(frd.SRs > 0.01)[0][0]
t0 = frd.Time[id0]


fig = plt.figure(figsize=(6.0, 7.5 * 5.0 / 16), dpi=80)
ax = fig.add_subplot(111)

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


print(id0)
plt.plot(frd.Time[id0:] - t0, frd.SRs[id0:], label="preferred model")
data_Latour = np.loadtxt("Latour.txt", delimiter=",")
plt.plot(
    data_Latour[:, 0],
    data_Latour[:, 1],
    label="Latour et al. (2025)",
    marker="o",
    markersize=4,
    linestyle="None",
    color="black",
)

plt.legend(frameon=False)
plt.xlim([0, 4])
plt.xlabel(f"Time (s) since {t0:.1f}s post-rupture onset")
# plt.xlabel(f"Time (seconds). time = 0 is {t0:.1f}s after rupture onset in the dynamic rupture model)")
plt.ylabel("Slip rate along strike (m/s)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()


fn = "CCTV_comparison.svg"
plt.savefig(fn)

plt.show()
