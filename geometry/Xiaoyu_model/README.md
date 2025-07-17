# CAD and Mesh Generation

1. Generate vtk and stl files from the slip and vert ascii data describing the finite fault model.

```
python3 convert_Xiaoyu_model.py outdated_model_for_illustration
```

2. Generate ASAGI file for the DR workflow.
Create a structured grid from the VTK to be used with SeisSol's ASAGI input format.
```
python3 interpolate_regular_grid.py 
```
Add the resulting .nc file to the `custom_setup_files` section in your DR config YAML.

3.  Refine and smooth the fault surface mesh.

```
python ~/SeisSol/Meshing/creating_geometric_models/refine_and_smooth_mesh.py mesh.stl --P 4 --N 0 --fix_boundary
```

4. Process topography and generate simulation domain box.

```
gebco=GEBCO_05_Jun_2025_14bff4208ba0/gebco_2024_n24.6_s16.0_w93.0_e99.5.nc
proj="+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.92 +lat_0=22.00"
ds=1
python ~/SeisSol/Meshing/creating_geometric_models/create_surface_from_rectilinear_grid.py $gebco gebco_2024_n24.6_s16.0_w93.0_e99.5_ds$sub.ts --sub $ds --proj "$proj"
python ~/SeisSol/Meshing/creating_geometric_models/generate_box.py --rangeFromTopo $gebco --zdim " -200e3" 10e3 box.stl --proj "$proj" --shrink 0.95
```

5. Assemble CAD and generate mesh in SimModeler.
