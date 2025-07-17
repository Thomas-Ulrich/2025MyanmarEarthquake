import numpy as np
import pyvista as pv
from pyproj import Transformer
import seissolxdmfwriter as sxw
from scipy.spatial import cKDTree
import argparse
import os


def clean_delaunay(points, triangles):
    def triangle_area_3d(tri_pts):
        a, b, c = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
        ab = b - a
        ac = c - a
        cross = np.cross(ab, ac)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        return area

    def min_edge_lengths(tri_pts):
        a, b, c = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
        ab = np.linalg.norm(b - a, axis=1)
        bc = np.linalg.norm(c - b, axis=1)
        ca = np.linalg.norm(a - c, axis=1)
        return np.minimum.reduce([ab, bc, ca])

    def triangle_aspect_ratio(tri_pts):
        a = np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 0], axis=1)
        b = np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 1], axis=1)
        c = np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 2], axis=1)

        s = 0.5 * (a + b + c)
        area = 0.5 * np.linalg.norm(
            np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]),
            axis=1,
        )

        inradius = area / s
        longest = np.maximum.reduce([a, b, c])
        ar = longest / (2 * inradius + 1e-16)  # avoid div-by-zero

        return ar

    tri_pts = points[triangles]
    areas = triangle_area_3d(tri_pts)
    AR = triangle_aspect_ratio(tri_pts)
    triangles = triangles[(areas > 0.01) & (AR < 4)]
    return triangles


def read_vert_slip(folder):
    if folder != "model3simple":
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
        triangles = np.array(triangles)
    slip = np.loadtxt(f"{folder}/slip.txt", delimiter=",")
    if folder == "model3simple":
        from sklearn.decomposition import PCA
        from scipy.spatial import Delaunay

        points = slip[:, :3].copy()
        # points[:, 2] /= 1e3
        myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.92 +lat_0=22.00"
        transformer = Transformer.from_crs("epsg:4326", myproj, always_xy=True)
        points[:, 0], points[:, 1] = transformer.transform(points[:, 0], points[:, 1])
        # print(points)
        pca = PCA(n_components=2)
        xyz = pca.fit_transform(points)
        triangles = Delaunay(xyz).simplices
        triangles = clean_delaunay(points, triangles)

    return points, triangles, slip


def read_slip(fn):
    # Load slip data
    slip = np.loadtxt(f"{folder}/slip.txt", delimiter=",")
    if len(slip) == len(triangles):
        cell_data = True
    elif len(slip) == len(points):
        cell_data = False
    else:
        raise ValueError(
            f"Mismatch between points and slip data: slip {len(slip)} pts: {len(points)} triangles: {len(triangles)}"
        )
    return slip, cell_data


def is_cell_data(points, triangles, slip):
    if len(slip) == len(triangles):
        cell_data = True
    elif len(slip) == len(points):
        cell_data = False
    else:
        raise ValueError(
            f"Mismatch between points and slip data: slip {len(slip)} pts: {len(points)} triangles: {len(triangles)}"
        )
    return cell_data


def generate_grid(points, triangles, slip):
    myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.92 +lat_0=22.00"
    transformer = Transformer.from_crs("epsg:4326", myproj, always_xy=True)
    if folder != "model3simple":
        points[:, 0], points[:, 1] = transformer.transform(points[:, 0], points[:, 1])
        points[:, 2] *= 1e3

    # Convert triangles to pyvista format: [3, pt0, pt1, pt2] per triangle
    cells = np.hstack([np.full((len(triangles), 1), 3), triangles]).astype(np.int32)
    cell_types = np.full(len(triangles), pv.CellType.TRIANGLE, dtype=np.uint8)

    # Create pyvista unstructured grid
    grid = pv.UnstructuredGrid(cells, cell_types, points)

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
        assert np.max(dist) < 0.1, dist

        # Assign slip to point data
        grid.point_data["sls"] = sls[idx]
        grid.point_data["sld"] = sld[idx]

    # Plotting
    # grid.plot(scalars="sls", show_edges=True, cmap="viridis" if cell_data else "viridis_r")
    # grid.plot(scalars="sld", show_edges=True, cmap="viridis" if cell_data else "viridis_r")
    return grid


parser = argparse.ArgumentParser(description="convert Xiaoyu model to vtk")
parser.add_argument(
    "folder",
    help="path to folder containing vert.txt and slip.txt",
)
args = parser.parse_args()
folder = args.folder
znew0 = 500.0


fn = f"{folder}/vert.txt"
if os.path.exists(fn):
    points, triangles, slip = read_vert_slip(folder)
else:
    data = np.loadtxt(f"{folder}/slip_model_altered.txt", delimiter=",", skiprows=4)
    # print(data.shape)
    layer_id = data[:, 2]

    data_layer1 = data[layer_id == 1, :]
    slip = data_layer1[:, 3:5]
    # Parse vertices: each line = 3 points (3 coords each) → 9 floats per line
    triangles = []
    points = []
    point_index = {}

    for i, row in enumerate(data_layer1):
        coords = row[18:]
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
    triangles = np.array(triangles)

    data_other_layers = data[layer_id != 1, :]
    slip = data_other_layers[:, 3:5]
    triangles = []
    points = []
    point_index = {}

    for i, row in enumerate(data_layer1):
        coords = row[18:]
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
    triangles = np.array(triangles)


cell_data = is_cell_data(points, triangles, slip)

grid = generate_grid(points, triangles, slip)
"""
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
"""

grid.points[grid.points[:, 2] == 0, 2] = znew0
if cell_data:
    grid.cell_data["ASl"] = np.sqrt(
        grid.cell_data["sls"] ** 2 + grid.cell_data["sld"] ** 2
    )
else:
    grid.point_data["ASl"] = np.sqrt(
        grid.point_data["sls"] ** 2 + grid.point_data["sld"] ** 2
    )
grid.save("fault_slip.vtk")
print("done writing fault_slip.vtk")
# Save geometry as STL
surface = grid.extract_surface().triangulate()
surface.save("mesh.stl")
print("done writing mesh.stl")
