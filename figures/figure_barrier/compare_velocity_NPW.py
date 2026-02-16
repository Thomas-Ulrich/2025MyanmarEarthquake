import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.signal.cross_correlation import correlate, xcorr_max
from pyproj import Transformer

ps = 14
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


# --- Parameters ---
fmin = 0.05
fmax = 0.5
t_before = 30  # Changed to positive for easier logic
t_after = 110
network = "GE"
station = "NPW"


def readSeisSolReceiver(file):
    """READ SeisSol receiver"""
    with open(file) as fid:
        fid.readline()
        variablelist = fid.readline()[11:].split(",")
        variablelist = np.array([a.strip().strip('"') for a in variablelist])
        xsyn = float(fid.readline().split()[2])
        ysyn = float(fid.readline().split()[2])
        zsyn = float(fid.readline().split()[2])
        synth = np.loadtxt(fid)
        selected_vars = ["Time", "v1", "v2", "v3"]
        indices = [i for i, var in enumerate(variablelist) if var in selected_vars]
        synth = synth[:, indices]
    return ([xsyn, ysyn, zsyn], synth)


def matchStation2Receiver(receiver_coords, station_coords, myproj):
    transformer = Transformer.from_crs(myproj, "epsg:4326", always_xy=True)
    lon, lat, depth = transformer.transform(
        receiver_coords[0], receiver_coords[1], receiver_coords[2]
    )
    for station_name, coordstat in station_coords.items():
        if (abs(lon - coordstat[0]) < 4e-2) and (abs(lat - coordstat[1]) < 4e-2):
            print(station_name, coordstat)
            return station_name
    return []


def retrieve_waveform():
    client = Client("GEOFON")

    # Define station and time window
    network = "GE"
    station = "NPW"
    location = "*"
    channel = "HNN"

    starttime = UTCDateTime("2025-03-28T06:20:52.715Z")
    endtime = starttime + 200

    # Try downloading waveform
    st = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=starttime,
        endtime=endtime,
    )

    # Optional: attach instrument response to remove it later
    inv = client.get_stations(
        network=network,
        station=station,
        location=location,
        channel=channel,
        level="response",
    )
    st.attach_response(inv)
    # st.detrend(type="linear")
    st.remove_response(output="ACC")
    st.taper(max_percentage=10.0 / 300.0)

    st.integrate()
    st.detrend(type="linear")
    st.detrend(type="demean")  # Remove mean
    for tr in st:
        tr.data = tr.data - tr.data[0]
    return st, starttime


def get_seismic_shift(tr_obs, tr_syn, t_start=40, t_end=80):
    """
    Calculates the time shift between observation and synthetic.
    Works with older ObsPy versions.
    """
    obs = tr_obs.copy()
    syn = tr_syn.copy()

    # 1. Resample synthetic to match observation
    if syn.stats.sampling_rate != obs.stats.sampling_rate:
        syn.resample(obs.stats.sampling_rate)

    # 2. Trim to the window of interest
    # We ensure they are the same length for the correlation
    t1 = obs.stats.starttime + t_start
    t2 = obs.stats.starttime + t_end
    obs.trim(t1, t2)

    # For synthetic, we assume starttime is 0
    syn.trim(syn.stats.starttime + t_start, syn.stats.starttime + t_end)
    # 3. Perform Cross-Correlation
    # In older versions, use positional arguments only
    cc = correlate(obs.data, syn.data, len(obs.data))

    # 4. Find the shift
    # shift_samples is the index relative to the center of the cc array
    shift_samples, max_coeff = xcorr_max(cc)

    # Convert samples to seconds
    shift_time = shift_samples / obs.stats.sampling_rate

    return shift_time, max_coeff


# --- Execution ---
myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.93 +lat_0=22.00"
station_coords = {"NPW": (96.14, 19.78)}

fn = f"GE.NPW_unfiltered_velocity.mseed"
if os.path.exists(fn):
    st = read(fn)
    origin_time = UTCDateTime("2025-03-28T06:20:52.715Z")
else:
    st, origin_time = retrieve_waveform()
    st.write(fn, format="MSEED")

st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

# Prepare Plot
#fig, ax = plt.subplots(1, 1, figsize=(7.5, 3))
fig, ax = plt.subplots(1, 1, figsize=(8.5, 3))

# 1. Process and Plot SeisSol (Synthetics)

prefix_path_list = [
    "../seissol_outputs/dyn_0080_coh0.25_1.0_B0.95_C0.15_R0.95",
    "seissol_output/dyn_0280_coh0.25_1.0_B0.95_C0.15_R0.95_barrier_dc",
]

for prefix_path in prefix_path_list:
    files = glob.glob(f"{prefix_path}-r*")
    for fname in files[::-1]:
        coords, synth = readSeisSolReceiver(fname)
        sta2comp = matchStation2Receiver(coords, station_coords, myproj)

        if sta2comp == "NPW":
            print(fname)
            time_synth = synth[:, 0]
            # v1=E, v2=N, v3=Z. We want N (index 2 in the synth array)
            vel_n_synth = synth[:, 2]

            # We MUST filter synthetics the same way as observations
            from obspy.core.trace import Trace

            tr_syn = Trace(
                data=vel_n_synth, header={"delta": time_synth[1] - time_synth[0]}
            )
            # tr_syn.taper(type="cosine", max_percentage=0.05)
            # tr_syn.detrend("demean")
            # tr_syn.detrend("linear")
            tr_syn.filter(
                "bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True
            )
            tr = st.select(component="N")[0]
            shift, score = get_seismic_shift(tr, tr_syn, t_start=40, t_end=80)
            print(shift, score)
            bn = os.path.basename(fname)
            if bn.startswith("dyn_0080"):
                label = "Preferred (at NPW"
                color = "blue"
                line_style = "-"
            else:
                label = "Barrier"
                color = "orange"

                if abs(coords[1] + 2.458418896780e05) < 1:
                    label += " (at NPW"
                    line_style = "-"
                    print(coords[1], label)
                elif coords[1] > -2.458418896780e05:
                    line_style = ":"
                    label += " (2 km North of NPW"
                    print(coords[1], label)
                else:
                    line_style = "--"
                    label += " (2 km South of NPW"
                    print(coords[1], label)
            label += f", {-shift:.1f}s shifted)"
            ax.plot(
                time_synth + shift,
                tr_syn.data,
                label=label,
                linewidth=1.5,
                color=color,
                linestyle=line_style,
            )

# 2. Process and Plot Observations
for tr in st.select(component="N"):
    # Synchronize time: t=0 is origin_time
    times = tr.times(reftime=origin_time)
    ax.plot(times, tr.data, "k", label="Observation")

# --- Formatting ---
ax.set_xlim(40, 80)  # Focus on the event duration
#ax.set_xlabel("Time since origin [s]")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity (m/s)")
ax.legend(loc="upper right", frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("NPW_velocity_comparison_N.svg")
#plt.show()
