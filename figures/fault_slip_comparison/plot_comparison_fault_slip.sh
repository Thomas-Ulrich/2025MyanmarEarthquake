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
output_prefix=$2

zoom=2.5
#view=normal-flip
view=normal-flip
#view=3D_view_myanmar_from_Southb.pvcc
rel=0.35
scalar_bar="0.91 0.35 160"
#win_size="800 2400"
win_size="2400 800"
# 600"
time="i-1"
contour_args="file_index=0 var=RT contour=black,4,0,max,5"
cp ../Figures_3_4/*.vtk .
mkdir output
light_quake_visualizer $file        --variable ASl              --cmap acton_r  --color_range "0 6.0"     --zoom $zoom --window $win_size --output ASl         --time $time --view $view --vtk_meshes "CCTV.vtk red 1;NPW.vtk blue 1"
light_quake_visualizer ../../geometry/fault_slip_m10.vtk        --variable ASl              --cmap acton_r  --color_range "0 6.0"     --zoom $zoom --window $win_size --output ASlref         --time $time --view normal  --vtk_meshes "CCTV.vtk red 1;NPW.vtk blue 1"

image_combiner --inputs output/ASlref.png output/ASl.png \
               --rel $rel 0.5 \
               --output output/${output_prefix}ASlcomp.png \
               --col 1

