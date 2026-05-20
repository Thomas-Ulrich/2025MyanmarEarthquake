!/usr/bin/env bash

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



zoom=3.2
#view=normal-flip
#view=normal
view=3D_view_myanmar_from_Southb.pvcc
rel=0.35
scalar_bar="0.91 0.35 160"
win_size="800 2400"
# 600"
time="i-1"
contour_args="file_index=0 var=RT contour=black,4,0,max,5"
vtk_meshes="CCTV.vtk red 1;NPW.vtk blue 1;nuc.vtk black 1"

mkdir output
# Slip, Peak slip rate, Vr, Vr_over_Vs, stress drop.
light_quake_visualizer $1        --variable mu_s           --cmap lapaz_r0  --color_range "0.15 0.75"       --zoom $zoom --window $win_size --output mu_s         --time $time --view $view --vtk_meshes "$vtk_meshes"
light_quake_visualizer $2        --variable ASl              --cmap davos_r0  --color_range "0 6"     --zoom $zoom --window $win_size --output ASl         --time $time --view $view --contour "$contour_args" --vtk_meshes "$vtk_meshes"

# Combine everything together
image_combiner --inputs output/mu_s.png output/ASl.png \
               --rel $rel 0.3 \
               --output output/Snapshots_barrier.png \
               --col 2

convert output/Snapshots_barrier.png -background white -alpha remove -alpha off output/Snapshots_barrier.png
