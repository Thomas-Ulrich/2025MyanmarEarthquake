mkdir -p folder_colorbars
generate_color_bar lipari_r --crange 0.1 0.85 --labelfont 12 --height 1.4 3 --nticks 3 --hor
generate_color_bar davos_r --crange 0.35 0.65 --labelfont 12 --height 1.4 3 --nticks 3 --hor
generate_color_bar davos_r --crange 0 35 --labelfont 12 --height 1.4 3 --nticks 3 --hor
generate_color_bar lipari_r --crange 0 9 --labelfont 12 --height 1.4 3 --nticks 3 --hor
generate_color_bar batlowW_r --crange -0.7 1 --labelfont 12 --height 1.4 3 --nticks 3 --hor
mv colorbar* folder_colorbars
