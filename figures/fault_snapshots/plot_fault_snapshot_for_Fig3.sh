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
fileSR=$2
output_prefix=$3

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

# Slip, Peak slip rate, Vr, Vr_over_Vs, stress drop.
light_quake_visualizer $file        --variable ASl              --cmap davos_r0  --color_range "0 6.0"     --zoom $zoom --window $win_size --output ASl         --time $time --view $view --contour "$contour_args" --vtk_meshes "$vtk_meshes"
light_quake_visualizer $file        --variable PSR              --cmap lipari_r0 --color_range "0 4.0"     --zoom $zoom --window $win_size --output PSR         --time $time --view $view --vtk_meshes "$vtk_meshes"
light_quake_visualizer $file        --variable Vr_kms           --cmap lapaz_r0  --color_range "0 6"       --zoom $zoom --window $win_size --output Vr          --time $time --view $view --vtk_meshes "$vtk_meshes"
light_quake_visualizer vr_over_vs.xdmf        --variable vr_over_vs  --cmap roma_r  --color_range "0.0 2"  --zoom $zoom --window $win_size --output Vr_over_vs          --time $time --view $view --vtk_meshes "$vtk_meshes"
light_quake_visualizer $file        --variable shear_stress_MPa           --cmap lapaz_r0  --color_range "0 9"       --zoom $zoom --window $win_size --output T_s          --time $time --view $view --vtk_meshes "$vtk_meshes"

# Slip rate
light_quake_visualizer $fileSR --variable SR --cmap magma_r0  --color_range "0 2.0" --zoom $zoom --window $win_size --output SR{time:.1f} --time "5;10;15;30;50;75;85" --view $view --vtk_meshes "$vtk_meshes"

# Combine everything together
image_combiner --inputs output/ASl.png output/Vr.png output/Vr_over_vs.png output/T_s.png output/SR5.0.png output/SR10.0.png output/SR15.0.png output/SR30.0.png output/SR50.0.png output/SR75.0.png output/SR85.0.png \
               --rel $rel 0.3 \
               --output output/${output_prefix}SnapshotsforFigure3.png \
               --col 11

