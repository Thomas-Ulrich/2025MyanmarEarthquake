#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 22:33:34 2025
"""

import obspy
from obspy import UTCDateTime
from obspy.core.inventory import Inventory, Network, Station, Site
from obspy.clients.fdsn import Client, RoutingClient
import os
from pyproj import Transformer
import matplotlib.pyplot as plt
# import groundMotionRoutines as gmr
import numpy as np
from obspy import read
import matplotlib
import glob
from scipy.integrate import cumulative_trapezoid
import argparse

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)

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

def matchStation2Receiver(receiver_coords,station_coords):
    """"Retrieve the station corresponding to a given SeisSol receiver"""   
    transformer = Transformer.from_crs(myproj, "epsg:4326", always_xy=True)
    lon,lat,depth = transformer.transform(receiver_coords[0], receiver_coords[1], receiver_coords[2])   
    sta2comp = []
    # print(receiver_coords[0], receiver_coords[1], receiver_coords[2])
    # print(lon,lat)
    for station, coordstat in station_coords.items():
        if (abs(lon - coordstat[0]) < 5e-2) & (abs(lat - coordstat[1]) < 5e-2):
            #print(f"Found matching station: {station}")      
            sta2comp = station
    return sta2comp


# Earthquake time

parser = argparse.ArgumentParser(description="Compare model synthetics with NPW station observations")
parser.add_argument(
    "pathObservations",
    help="Path to the observations"
)

parser.add_argument(
    "ensemble_dir",
    help="Path to the dynamic rupture model ensemble"
)

parser.add_argument(
    "best_model",
    help="Patteern for best model ((e.g. \"dyn_0073\")"
)

args = parser.parse_args()

eq_time= UTCDateTime("2025-03-28T06:20:52.715Z")
endtime = eq_time + 200  
tplot_max = 100.0

# setting up projections
lla = "epsg:4326"
myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.93 +lat_0=22.00"

# Read observations 
#pathObservations = "strong_motion_data"

fname = f"{args.pathObservations}/GE.NPW_HN_displacement_{eq_time.date}.mseed"
obs_trace = read(fname)

time_obs =  obs_trace.select(component='N')[0].times(reftime=eq_time)
data_obs = obs_trace.select(component='N')[0].data   
    
station_coords = {}
station_coords['NPW'] = (96.14,19.78)

# Read SeisSol receivers
#ensemble_dir = "receivers"

pattern = f"{args.ensemble_dir}/*dyn_*-receiver-*"
files = glob.glob(pattern)

fig = plt.figure(figsize=(5, 2), dpi=80)

ax = fig.add_subplot(111)



# Find all models
models = sorted(set(f.split("-receiver")[0] for f in files))

# Loop on the models
for model in models:
    print(model)
    
    # Find the reicever files 
    model_files = sorted(f for f in files if f.startswith(model + "-receiver"))
    
    # Loop on the receiver files (to find the one that match NPW stations)
    for fname in model_files:
         
        # Read SeisSol receiver
        coords, synth = readSeisSolReceiver(fname)
        
        # Find the corresponding station
        sta2comp = matchStation2Receiver(coords, station_coords)
        if len(sta2comp) == 0:
            # print(f"Station not found in the metadata file for receiver")
            continue
        else:
            print(f"Found matching station for receiver: {sta2comp}")
            ax.plot(time_obs,data_obs,'k')
            
            time_synth = synth[:,0]
            data_synth = synth[:,2]
            data_synth = cumulative_trapezoid(data_synth, time_synth, initial=0)  
            ax.plot(time_synth,data_synth,'#edeeeeff')

# Plot observation with uncertainty
ax.plot(time_obs,data_obs,'k',linewidth=1)
ax.plot(time_obs+0.8,data_obs,'--k',linewidth=0.5)
ax.plot(time_obs-0.8,data_obs,'--k',linewidth=0.5)



# Plot best model only
model_files = glob.glob(f"{args.ensemble_dir}/*{args.best_model}*-receiver-*.dat")

if not model_files:
    raise FileNotFoundError(f"No file found matching {args.best_model}")
    
    # Loop on the receiver files (to find the one that match NPW stations)
for fname in model_files:
     
    # Read SeisSol receiver
    coords, synth = readSeisSolReceiver(fname)
    
    # Find the corresponding station
    sta2comp = matchStation2Receiver(coords, station_coords)
    if len(sta2comp) == 0:
        # print(f"Station not found in the metadata file for receiver")
        continue
    else:
        #print(f"Found matching station for receiver: {sta2comp}")
        ax.plot(time_obs,data_obs,'k')
        
        time_synth = synth[:,0]
        data_synth = synth[:,2]
        data_synth = cumulative_trapezoid(data_synth, time_synth, initial=0)  
        ax.plot(time_synth,data_synth,color='blue',linewidth=2)
plt.xlim([0, 100])
plt.ylabel("NPW NS displacement (m)")
plt.xlabel("Time (s)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


fn = "figures/NPW_comparison.svg"
plt.savefig(fn, bbox_inches="tight")
print(f"done writing {fn}")

