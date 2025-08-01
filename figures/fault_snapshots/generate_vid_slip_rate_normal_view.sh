#!/bin/bash
file="$1"
prefix="$2"

echo "file: $file"
echo "prefix: $prefix"

view=normal-flip
rm output/SR*.png

light_quake_visualizer $file --variable SR --cmap magma_r0 --color_range "0 2" \
    --zoom 3.5 --window 2500 600 --annotate_time "k 0.15 0.6" --time "i:" \
    --scalar_bar "0.93 0.35 160" --view "$view" --font_size 20 --output SR%d --vtk_meshes "CCTV.vtk red 1;NPW.vtk blue 1"

ffmpeg -y -framerate 10 -i output/SR_%d.png \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -f mp4 -vcodec libx264 \
       -pix_fmt yuv420p -q:v 1 "Mynamar_${prefix}_SR.mp4"
ffmpeg -i "Mynamar_${prefix}_SR.mp4" -vf "fps=10,scale=2500:-1:flags=lanczos" "Mynamar_${prefix}_SR.gif"
