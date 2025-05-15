
#!/bin/bash
set -euo pipefail


# Prompt for user input
echo "Do you want to rerun the auto_model (y/n)"
read -r rerun_auto

echo "Do you want to rerun the teleseismic inversion (y/n)"
read -r rerun_tele

if [[ "$rerun_auto" == "y" ]]; then
   #we first run a auto inversion
    wasp model run $(pwd) auto_model -g data/cmtsolution -t body -t surf -d data/Teleseismic_Data/
    cp -r  20250328062054/ffm.0/NP2/ new_geom
fi

if [[ "$rerun_tele" == "y" ]]; then
    cd new_geom
    ../generate_jsons_segments_model_space.py
    wasp manage update-inputs $(pwd) -p -m -a
    wasp model run $(pwd) manual_model_add_data
   cp Solucion.txt plots
   cp modelling_summary.txt plots
fi
