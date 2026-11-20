"""Generate the tetrahedral slab meshes used by the Niederer benchmark."""

import argparse
import errno
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory


CASES_DIR = Path(__file__).resolve().parents[1] / "cases"
MESH_SIZES = (0.5, 0.2, 0.1)
CUBE_EDGE = 1.5
MESH_FILES = ("mesh.vtu", "domain.dat")


def tetrahedra(characteristic_length):
    import gmsh
    import numpy as np
    import pygmsh

    # Fragment the slab so the stimulus cube has a conforming interface.
    with pygmsh.occ.Geometry() as geometry:
        gmsh.option.setNumber("General.Terminal", 0)
        slab = geometry.add_box([0, 0, 0], [20, 7, 3])
        cube = geometry.add_box([0, 0, 0], [CUBE_EDGE] * 3)
        geometry.boolean_fragments(slab, cube)
        geometry.characteristic_length_min = characteristic_length
        geometry.characteristic_length_max = characteristic_length
        mesh = geometry.generate_mesh(dim=3)

    cells = mesh.get_cells_type("tetra")
    if not cells.size:
        raise ValueError("Gmsh produced no tetrahedra")
    used = np.unique(cells)
    indices = np.empty(len(mesh.points), dtype=np.int64)
    indices[used] = np.arange(len(used))
    return mesh.points[used], indices[cells]


def mean_edge_length(points, cells):
    import numpy as np

    # Encode each undirected edge as one integer to reduce peak memory use.
    count = len(points)
    edges = np.empty((6, len(cells)), dtype=np.int64)
    for row, (i, j) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
        edges[row] = np.minimum(cells[:, i], cells[:, j]) * count + np.maximum(
            cells[:, i], cells[:, j]
        )
    unique = np.unique(edges)
    displacement = points[unique // count] - points[unique % count]
    return float(np.linalg.norm(displacement, axis=1).mean())


def write_mesh(directory, points, cells):
    import numpy as np
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.util.vtkConstants import VTK_TETRA
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

    centers = points[cells].mean(axis=1)
    stimulated = np.all((centers >= -1e-12) & (centers < CUBE_EDGE - 1e-12), axis=1)
    domain = np.where(stimulated, 1, 2).astype(np.int32)
    if not stimulated.any() or stimulated.all():
        raise ValueError("Mesh must contain both stimulated and unstimulated cells")

    grid_points = vtkPoints()
    grid_points.SetData(numpy_to_vtk(points, deep=True))
    offsets = np.arange(0, cells.size + 1, 4, dtype=np.int64)
    grid_cells = vtkCellArray()
    grid_cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(cells.ravel().astype(np.int64, copy=False), deep=True),
    )
    grid = vtkUnstructuredGrid()
    grid.SetPoints(grid_points)
    grid.SetCells(VTK_TETRA, grid_cells)
    for data, name, values in (
        (grid.GetPointData(), "GlobalNodeID", np.arange(1, len(points) + 1, dtype=np.int32)),
        (grid.GetCellData(), "GlobalElementID", np.arange(1, len(cells) + 1, dtype=np.int32)),
        (grid.GetCellData(), "Domain", domain),
    ):
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        data.AddArray(array)

    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(directory / "mesh.vtu"))
    writer.SetInputData(grid)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    if writer.Write() != 1:
        raise OSError(f"Could not write mesh in {directory}")
    np.savetxt(directory / "domain.dat", domain, fmt="%d")


def generate_mesh(size, case_dir):
    """Materialize the mesh and domain files inside the chosen case."""
    directory = case_dir / "mesh"
    if directory.exists():
        if all((directory / name).is_file() for name in MESH_FILES):
            return
        raise ValueError(f"Incomplete generated mesh in {directory}. Move it before retrying.")

    existing = next(
        (
            mesh
            for mesh in sorted(CASES_DIR.glob(f"slab_h{size:g}_dt*/mesh"))
            if all((mesh / name).is_file() for name in MESH_FILES)
        ),
        None,
    )

    # Publish both files together. A simultaneous run can reuse the winning pair.
    with TemporaryDirectory(prefix=".mesh-", dir=case_dir) as temp:
        staged = Path(temp) / "mesh"
        staged.mkdir()
        if existing is None:
            characteristic_length = size
            for attempt in range(1, 5):
                points, cells = tetrahedra(characteristic_length)
                mean = mean_edge_length(points, cells)
                print(f"h={size:g} mm, iteration {attempt}, mean edge {mean:.6g} mm", flush=True)
                if abs(mean - size) <= 0.02 * size:
                    break
                characteristic_length *= size / mean
            else:
                raise ValueError(f"Mean edge {mean:.6g} mm is more than 2% from h={size:g} mm")
            write_mesh(staged, points, cells)
        else:
            for name in MESH_FILES:
                copyfile(existing / name, staged / name)
        try:
            staged.rename(directory)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not all(
                (directory / name).is_file() for name in MESH_FILES
            ):
                raise


def main():
    cases = {
        manifest.parent.name: size
        for size in MESH_SIZES
        for manifest in sorted(CASES_DIR.glob(f"slab_h{size:g}_dt*/benchmark.toml"))
    }
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--mesh", type=float, choices=MESH_SIZES, help="generate this size in mm")
    selection.add_argument("--case", choices=cases, help="generate only for this case")
    args = parser.parse_args()
    for name, size in cases.items():
        if (args.mesh is None or size == args.mesh) and (args.case is None or name == args.case):
            generate_mesh(size, CASES_DIR / name)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError) as error:
        sys.exit(f"Niederer mesh generation: {error}")
