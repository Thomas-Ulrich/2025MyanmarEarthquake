import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from asagiwriter import writeNetcdf
import numpy as np


def project_to_yz_plane(mesh, x_fixed=0.0):
    projected = mesh.copy()
    projected.points[:, 0] = x_fixed  # set x to constant value
    return projected


def create_yz_probe_grid(surface, spacing_y, spacing_z, x_fixed=0.0):
    _, _, y_min, y_max, z_min, z_max = surface.GetBounds()

    ny = int((y_max - y_min) / spacing_y) + 1
    nz = int((z_max - z_min) / spacing_z) + 1

    grid = vtk.vtkImageData()
    grid.SetOrigin(x_fixed, y_min, z_min)
    grid.SetSpacing(1.0, spacing_y, spacing_z)  # x-spacing doesn't matter
    grid.SetDimensions(1, ny, nz)
    y_coords = np.linspace(y_min, y_min + (ny - 1) * spacing_y, ny)
    z_coords = np.linspace(z_max - (nz - 1) * spacing_z, z_max, nz)

    return y_coords, z_coords, grid


def probe(surface, grid):
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetSourceData(surface)
    probe_filter.SetInputData(grid)
    probe_filter.Update()
    return probe_filter.GetOutput()


def extract_numpy_array(image_data, name):
    arr = image_data.GetPointData().GetArray(name)
    if arr is None:
        raise ValueError(f"Array '{name}' not found in point data.")
    np_arr = vtk_to_numpy(arr)
    dims = image_data.GetDimensions()  # (1, ny, nz)
    return np_arr.reshape((dims[1], dims[2]))


surface = pv.read("fault_slip.vtk")  # or any other vtk/obj/stl file
# Project onto yz-plane at x = 0.0
yz_surface = project_to_yz_plane(surface)
y, z, grid = create_yz_probe_grid(yz_surface, spacing_y=4e3, spacing_z=2e3, x_fixed=0.0)

probed_output = probe(yz_surface, grid)
out = {}
out["strike_slip"] = extract_numpy_array(probed_output, "sls")
out["dip_slip"] = extract_numpy_array(probed_output, "sld")
ny, nz = out["strike_slip"].shape

for var in ["strike_slip", "dip_slip"]:
    out[var] = out[var].reshape(nz, ny)

for var in ["rupture_onset", "effective_rise_time", "acc_time"]:
    out[var] = 0.0 * out["strike_slip"]
out["rake_interp_low_slip"] = 0.0 * out["strike_slip"] + 180


prefix = "model"
variables = [
    "strike_slip",
    "dip_slip",
    "rupture_onset",
    "effective_rise_time",
    "acc_time",
    "rake_interp_low_slip",
]

for i, var in enumerate(variables):
    writeNetcdf(
        f"{prefix}_{var}",
        [y, z],
        [var],
        [out[var]],
        paraview_readable=True,
    )

writeNetcdf(
    f"{prefix}_ASAGI",
    [y, z],
    variables,
    [out[v] for v in variables],
    paraview_readable=False,
)
