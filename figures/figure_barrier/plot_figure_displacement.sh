#python ~/repositories/TuSeisSolScripts/onHdf5/apply_operation_on_seissol_data.py seissol_output_large/dyn_1080_coh0.25_1.0_B0.95_C0.15_R0.95_barrier_last-surface.xdmf seissol_output_large/dyn_0080_coh0.25_1.0_B0.95_C0.15_R0.95_last-surface.xdmf  --var u2

barrier_file=seissol_output_large/dyn_1080_coh0.25_1.0_B0.95_C0.15_R0.95_barrier_last-surface.xdmf
diff_file=seissol_output_large/diff_dyn_1080_coh0.25_1.0_B0.95_C0.15_R0.95_barrier_last-surface.xdmf

light_quake_visualizer --view xy_myanmar.pvcc $barrier_file --window 600 1600 --var u2 --time "i-1" --cmap roma --color_range "-3 3" --zoom 7.5 --vtk_meshes "CCTV.vtk red 1;NPW.vtk blue 1;nuc_epi.vtk black 1;grid.vtk black 0.2;fault_trace.vtk darkblue 2.0" --output u2 --light 0.1 0.6 0.5

light_quake_visualizer --view xy_myanmar.pvcc $diff_file --window 600 1600 --var u2 --time "i-1" --cmap broc --color_range "-0.8 0.8" --zoom 7.5 --vtk_meshes "CCTV.vtk red 1;NPW.vtk blue 1;nuc_epi.vtk black 1;grid.vtk black 0.2;fault_trace.vtk darkblue 2.0" --output diff_u2 --light 0.1 0.6 0.5
