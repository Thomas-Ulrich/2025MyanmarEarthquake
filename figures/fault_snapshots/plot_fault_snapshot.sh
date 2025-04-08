#!/usr/bin/env bash

# Just a small check for the version of light_quake_visualizer
version_to_check=$(light_quake_visualizer --version | awk '{print $NF}')  # Get the version from the command
target_version="0.3.2"

# Compare versions using sort
if [[ $(echo -e "$version_to_check\n$target_version" | sort -V | head -n 1) == "$target_version" ]]; then
    # Do nothing if the version is sufficient
    :
else
    # Raise an error if the version is too low
    echo "Error: Version is too low. Please update to a version greater than $target_version."
    exit 1  # Exit with a non-zero status to indicate an error
fi



file=$1
file_param=$2
output_prefix=$3

zoom=3.2
view=normal
rel=0.35
scalar_bar="0.91 0.35 160"
win_size="2400 600"
time="i-1"
contour_args="file_index=0 var=RT contour=black,4,0,max,5"

light_quake_visualizer $file        --variable ASl              --cmap davos_r0  --color_range "0 6.0"     --zoom $zoom --window $win_size --output ASl         --time $time --view $view --scalar_bar "$scalar_bar" --contour "$contour_args"
light_quake_visualizer $file        --variable PSR              --cmap lipari_r0 --color_range "0 8.0"     --zoom $zoom --window $win_size --output PSR         --time $time --view $view --scalar_bar "$scalar_bar"
light_quake_visualizer $file        --variable Vr_kms           --cmap lapaz_r0  --color_range "0 6"       --zoom $zoom --window $win_size --output Vr          --time $time --view $view --scalar_bar "$scalar_bar"
light_quake_visualizer $file_param  --variable mu_s             --cmap davos_r   --color_range "0.2 0.6"   --zoom $zoom --window $win_size --output mus         --time $time --view $view --scalar_bar "$scalar_bar"
light_quake_visualizer $file_param  --variable d_c              --cmap lipari_r  --color_range "0.13 0.91" --zoom $zoom --window $win_size --output dc          --time $time --view $view --scalar_bar "$scalar_bar"
light_quake_visualizer $file_param  --variable shear_stress_MPa --cmap lapaz_r   --color_range "0.0 20.0"  --zoom $zoom --window $win_size --output shearstress --time $time --view $view --scalar_bar "$scalar_bar"

image_combiner --inputs output/mus.png output/dc.png output/shearstress.png output/ASl.png output/PSR.png output/Vr.png \
               --rel $rel 1.0 \
               --output output/${output_prefix}ASlVrPSR.png \
               --col 1

