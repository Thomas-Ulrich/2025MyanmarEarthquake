#!/bin/bash
set -euo pipefail

# Prompt for user input
echo "Do you want to download CMT data? (y/n)"
read -r download_cmt

echo "Do you want to download Teleseismic data? (y/n)"
read -r download_teleseismic

# Download CMT if needed
if [[ "$download_cmt" == "y" || "$download_cmt" == "Y" ]]; then
    echo "Downloading CMT data..."
    mkdir -p data
    curl https://ds.iris.edu/spudservice/momenttensor/23108833/cmtsolution -o data/cmtsolution
    scripts/modify_hypo_in_cmtsolution.py
else
    echo "Skipping CMT download."
fi


# Download Teleseismic data if needed
if [[ "$download_teleseismic" == "y" || "$download_teleseismic" == "Y" ]]; then
    echo "Downloading Teleseismic data..."
    mkdir -p data/Teleseismic_Data
    wasp manage acquire data/Teleseismic_Data data/cmtsolution -t body
else
    echo "Skipping Teleseismic download."
fi

