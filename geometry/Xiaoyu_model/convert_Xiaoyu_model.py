import numpy as np
import pyvista as pv
from pyproj import Transformer
import seissolxdmfwriter as sxw
from scipy.spatial import cKDTree
import argparse


parser = argparse.ArgumentParser(description="convert Xiaoyu model to vtk")
parser.add_argument(
    "folder",
    help="path to folder containing vert.txt and slip.txt",
)
args = parser.parse_args()
folder = args.folder
znew0 = 500.0

# Load vertex data
with open(f"{folder}/vert.txt", "r") as f:
    lines = f.readlines()

# Parse vertices: each line = 3 points (3 coords each) → 9 floats per line
triangles = []
points = []
point_index = {}

for line in lines:
    coords = list(map(float, line.strip().split(",")))
    assert len(coords) == 9, "Each line should have 3 points (9 values)"
    tri = []
    for i in range(0, 9, 3):
        point = tuple(coords[i : i + 3])
        if point not in point_index:
            point_index[point] = len(points)
            points.append(point)
        tri.append(point_index[point])
    triangles.append(tri)

points = np.array(points)
myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.92 +lat_0=22.00"
transformer = Transformer.from_crs("epsg:4326", myproj, always_xy=True)

points[:, 0], points[:, 1] = transformer.transform(points[:, 0], points[:, 1])
points[:, 2] *= 1e3

triangles = np.array(triangles)

# Convert triangles to pyvista format: [3, pt0, pt1, pt2] per triangle
cells = np.hstack([np.full((len(triangles), 1), 3), triangles]).astype(np.int32)
cell_types = np.full(len(triangles), pv.CellType.TRIANGLE, dtype=np.uint8)

# Create pyvista unstructured grid
grid = pv.UnstructuredGrid(cells, cell_types, points)

# Load slip data
slip = np.loadtxt(f"{folder}/slip.txt", delimiter=",")
if len(slip) == len(triangles):
    cell_data = True
elif len(slip) == len(points):
    cell_data = False
else:
    raise ValueError("Mismatch between points and slip data")

if cell_data:
    # Add slip as cell data
    grid.cell_data["sls"] = slip[:, 0]
    grid.cell_data["sld"] = slip[:, 1]
else:
    # Load slip file: lon, lat, depth (km), sls, sld
    slip_lonlat = slip[:, :2]
    slip_depth = slip[:, 2]
    sls = slip[:, 3]
    sld = slip[:, 4]

    # Project slip points
    slip_x, slip_y = transformer.transform(slip_lonlat[:, 0], slip_lonlat[:, 1])
    slip_z = slip_depth

    slip_points = np.column_stack([slip_x, slip_y, slip_z])

    # Match each mesh point to the closest slip point
    tree = cKDTree(slip_points)
    dist, idx = tree.query(points)

    # Optionally: check that distances are small enough
    assert np.max(dist) < 0.1

    # Assign slip to point data
    grid.point_data["sls"] = sls[idx]
    grid.point_data["sld"] = sld[idx]


# Plotting
# grid.plot(scalars="sls", show_edges=True, cmap="viridis" if cell_data else "viridis_r")
# grid.plot(scalars="sld", show_edges=True, cmap="viridis" if cell_data else "viridis_r")


if cell_data:
    cells = grid.cells.reshape((grid.cells.size // 4, 4))[:, 1:]

    sxw.write(
        "Xiaoyu_finite_fault_model",
        grid.points,
        cells,
        {
            "ASl": np.sqrt(slip[:, 0] ** 2 + slip[:, 1] ** 2),
            "Sls": slip[:, 0],
            "Sld": slip[:, 1],
        },
        {0.0: 0},
        reduce_precision=True,
    )
else:
    grid.points[grid.points[:, 2] == 0, 2] = znew0
    grid.save("fault_slip.vtk")
    # Save geometry as STL
    surface = grid.extract_surface().triangulate()
    surface.save("mesh.stl")
