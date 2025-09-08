from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
from obspy.signal.detrend import polynomial
import numpy as np
import glob
from pyproj import Transformer
import os


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
        # Extract only relevant columns
        selected_vars = ["Time", "v1", "v2", "v3"]
        indices = [i for i, var in enumerate(variablelist) if var in selected_vars]
        synth = synth[:, indices]  # Select columns matching "time", "v1", "v2", "v3"
    return ([xsyn, ysyn, zsyn], synth)


def matchStation2Receiver(receiver_coords, station_coords):
    """ "Retrieve the station corresponding to a given SeisSol receiver"""
    transformer = Transformer.from_crs(myproj, "epsg:4326", always_xy=True)
    lon, lat, depth = transformer.transform(
        receiver_coords[0], receiver_coords[1], receiver_coords[2]
    )
    sta2comp = []
    # print(receiver_coords[0], receiver_coords[1], receiver_coords[2])
    # print(lon,lat)
    for station, coordstat in station_coords.items():
        if (abs(lon - coordstat[0]) < 5e-2) & (abs(lat - coordstat[1]) < 5e-2):
            # print(f"Found matching station: {station}")
            sta2comp = station
    return sta2comp


def retrieve_waveform():
    client = Client("GEOFON")

    # Define station and time window
    network = "GE"
    station = "NPW"
    location = "*"
    channel = "HN*"

    starttime = UTCDateTime("2025-03-28T06:20:52.715Z") - t_before
    endtime = starttime + 300

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

    selected_band = channel[0:2]
    fname = f"{network}.{station}_{selected_band}_{kind_vd}_{starttime.date}.mseed"
    st.write(fname, format="MSEED")
    # st.integrate()
    return st


t_before = 100
t_after = 110
kind_vd = "acceleration"

fn = "GE.NPW_HN_acceleration_2025-03-28.mseed"
if os.path.exists(fn):
    st = read("GE.NPW_HN_acceleration_2025-03-28.mseed")
else:
    st = retrieve_waveform()

# setting up projections
lla = "epsg:4326"
myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.93 +lat_0=22.00"
station_coords = {}
station_coords["NPW"] = (96.14, 19.78)

files = glob.glob(
    "../figures/seissol_outputs/dyn_0080_coh0.25_1.0_B0.95_C0.15_R0.95-r*"
)
print(files)


# Loop on the receiver files (to find the one that match NPW stations)
for fname in files:
    # Read SeisSol receiver
    coords, synth = readSeisSolReceiver(fname)

    # Find the corresponding station
    sta2comp = matchStation2Receiver(coords, station_coords)
    if len(sta2comp) == 0:
        # print(f"Station not found in the metadata file for receiver")
        continue
    else:
        print(f"Found matching station for receiver: {sta2comp}")

        time_synth = synth[:, 0]
        data_synth = synth[:, 1:]
        data_synth = np.gradient(data_synth, time_synth, axis=0)

# Create a two-row plot
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
cols = ["k", "r", "b"]
lookup = {"E": 0, "N": 1, "Z": 2}
for k, tr in enumerate(st):
    data = tr.data
    t = np.arange(tr.stats.npts) * tr.stats.delta  # time vector

    mask = (t >= t_before) & (t <= t_after + t_before)
    t_sel = t[mask] - t_before  # time since t_before
    data_sel = data[mask]

    # Compute log amplitude
    log_amplitude = np.log10(np.abs(data_sel) + 1e-12)  # avoid log(0)
    ylim_min = -7  # corresponds to log10(1e-7)
    ylim_max = np.max(log_amplitude)
    from scipy.ndimage import gaussian_filter1d

    log_amplitude = gaussian_filter1d(log_amplitude, sigma=4.0)

    label = tr.id[-1]
    print(label)
    # Top: original time series
    axs[0].plot(t_sel, data_sel, color=cols[k], label=label)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title(f"Time Series - {kind_vd}")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Bottom: log amplitude
    axs[1].plot(t_sel, log_amplitude, color=cols[k], label=label)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Log10 Amplitude")
    axs[1].set_title(f"Observations")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].set_ylim(ylim_min, np.max(log_amplitude))

    # Bottom: log amplitude

    log_synth = np.log10(np.abs(data_synth[:, lookup[label]]) + 1e-12)
    log_synth = gaussian_filter1d(log_synth, sigma=4.0)
    axs[2].plot(time_synth, log_synth, color=cols[k])
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Log10 Amplitude")
    axs[2].set_title(f"Synthetics")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].set_ylim(ylim_min, np.max(log_amplitude))

axs[0].legend()
plt.tight_layout()
plt.show()
