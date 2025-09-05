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
view=normal-flip
#view=normal
#view=3D_view_myanmar_from_Southb.pvcc
rel=0.5
scalar_bar="0.91 0.35 160"
#win_size="800 2400"
win_size="2300 800"
# 600"
time="i-1"
contour_args="file_index=0 var=RT contour=black,4,0,max,5"
cp ../Figures_3_4/*.vtk .
mkdir output

#light_quake_visualizer ../seissol_outputs/fault_from_fl33_input_extracted.xdmf        --variable ASl              --cmap davos_r0  --color_range "0 6"     --zoom $zoom --window $win_size --output ASl         --time $time --view $view #--contour "$contour_args" #--scalar_bar "$scalar_bar" 
light_quake_visualizer ../seissol_outputs/R_ratio.xdmf        --variable d_c              --cmap lipari_r --color_range "0.1 1.4"     --zoom $zoom --window $win_size --output d_c         --time $time --view $view #--scalar_bar "$scalar_bar"
light_quake_visualizer ../seissol_outputs/R_ratio.xdmf        --variable mu_s             --cmap davos_r --color_range "0.35 0.65"     --zoom $zoom --window $win_size --output mu_s         --time $time --view $view #--scalar_bar "$scalar_bar"
light_quake_visualizer ../seissol_outputs/R_ratio.xdmf        --variable pos_pn0              --cmap davos_r0 --color_range "0 1.6e7"     --zoom $zoom --window $win_size --output pn0         --time $time --view $view #--scalar_bar "$scalar_bar"
light_quake_visualizer ../seissol_outputs/R_ratio.xdmf        --variable tau              --cmap lipari_r0 --color_range "0 9e6"     --zoom $zoom --window $win_size --output tau         --time $time --view $view #--scalar_bar "$scalar_bar"
light_quake_visualizer ../seissol_outputs/R_ratio.xdmf        --variable R_ratio           --cmap batlowW_r  --color_range "0 1.0"       --zoom $zoom --window $win_size --output Rratio          --time $time --view $view --vtk_meshes "CCTV.vtk red 1;NPW.vtk blue 1" #--scalar_bar "$scalar_bar"

image_combiner --inputs output/d_c.png output/mu_s.png  output/pn0.png output/tau.png output/Rratio.png \
               --rel $rel 1.0 \
               --output output/${output_prefix}_tau_Rratio.png \
               --col 1
convert output/${output_prefix}_tau_Rratio.png -background white -alpha remove -alpha off output/${output_prefix}_tau_Rratio.png
