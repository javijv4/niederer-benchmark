"""Generate the Niederer diagonal, P8 surface, and isochrone PDFs."""

import argparse
import json
from pathlib import Path
import sys
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkFiltersCore import vtkCutter, vtkProbeFilter
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

from benchmark import CASES_DIR
from generate_mesh import MESH_SIZES


TIME_STEPS = (0.005, 0.01, 0.05)
FIGURE_DT = TIME_STEPS[0]
SLICE_SIZES = (0.5, 0.1)
P1 = (0.0, 0.0, 0.0)
P8 = (20.0, 7.0, 3.0)
PUBLISHED_INTERVAL = (37.8, 48.7)
LINE_COLORS = {0.5: "blue", 0.2: "green", 0.1: "red"}


def results_dir(size, dt):
    return CASES_DIR / f"slab_h{size:g}_dt{dt:g}" / "results"


def load_activation(path):
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    values = mesh.GetPointData().GetArray("ActivationTime")
    if reader.GetErrorCode() or not mesh.GetNumberOfCells() or values is None:
        raise ValueError(f"Could not read an activation mesh from {path}")
    if not np.isfinite(vtk_to_numpy(values)).all():
        raise ValueError(f"Nonfinite activation times in {path}")
    return mesh


def solver_caption(case_builds):
    groups = {}
    for (size, dt), build in case_builds.items():
        identity = (build["version"], build["tag"], build["commit"]) if build else None
        groups.setdefault(identity, []).append(f"h={size:g}, dt={dt:g}")

    captions = []
    for identity, cases in groups.items():
        if identity is None:
            caption = "svMultiPhysics build unavailable"
        else:
            version, _tag, commit = identity
            caption = f"svMultiPhysics {version}\nCommit {commit[:8]}"
        if len(groups) > 1:
            caption += "\n" + fill("  |  ".join(cases), width=48)
        captions.append(caption)
    return "\n\n".join(captions)


def save_figure(fig, path, case_builds):
    caption = solver_caption(case_builds)
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=6)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                metadata={"Subject": " | ".join(caption.splitlines())})
    plt.close(fig)
    print(f"Wrote {path}", flush=True)


def figure_size(width_cm, height_cm):
    return width_cm / 2.54, height_cm / 2.54


def sample_diagonal(mesh):
    line = vtkLineSource()
    line.SetPoint1(P1)
    line.SetPoint2(P8)
    line.SetResolution(400)
    probe = vtkProbeFilter()
    probe.SetInputConnection(line.GetOutputPort())
    probe.SetSourceData(mesh)
    probe.Update()
    sampled = probe.GetOutput()
    valid = vtk_to_numpy(sampled.GetPointData().GetArray("vtkValidPointMask"))
    if not valid.all():
        raise ValueError("The activation mesh does not cover the entire P1–P8 diagonal")
    points = vtk_to_numpy(sampled.GetPoints().GetData())
    values = vtk_to_numpy(sampled.GetPointData().GetArray("ActivationTime"))
    return np.linalg.norm(points - P1, axis=1), values


def draw_diagonal(curves, output_dir, builds):
    fig, ax = plt.subplots(figsize=figure_size(6.0, 4.6), layout="constrained")
    ax.axhspan(*PUBLISHED_INTERVAL, color="0.75", alpha=0.35, zorder=0, label="Niederer et al.")
    for size, (distance, activation) in curves.items():
        ax.plot(distance, activation, color=LINE_COLORS[size], linewidth=1.2,
                label=rf"$\Delta x = {size:g}$ mm")
    ax.set(xlim=(0, 21.4), ylim=(0, 50), xticks=(0, 10.7, 21.4),
           yticks=(0, 10, 20, 30, 40, 50), xlabel="distance (mm)", ylabel="activation time (ms)")
    ax.grid(True, color="0.85", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.6, length=3, pad=2)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, output_dir / "benchmark_line.pdf",
                {(size, FIGURE_DT): builds.get((size, FIGURE_DT)) for size in curves})


def draw_surface(surface, output_dir, builds):
    fig, ax = plt.subplots(figsize=figure_size(6.0, 5.2),
                           subplot_kw={"projection": "3d"}, layout="constrained")
    dx_index = np.arange(len(MESH_SIZES))
    dt_index = np.arange(len(TIME_STEPS))
    x, t = np.meshgrid(dx_index, dt_index, indexing="ij")
    if np.isfinite(surface).all():
        ax.plot_surface(x, t, surface, cmap="Blues", vmin=0, vmax=150, edgecolor="0.25",
                        linewidth=0.4, antialiased=True, shade=False)
    else:
        # Keep individual runs visible when adjacent grid points are missing.
        available = np.isfinite(surface)
        ax.scatter(x[available], t[available], surface[available], color="tab:blue", s=10)
    ax.set_xlabel(r"$\Delta x$ (mm)", labelpad=6)
    ax.set_ylabel(r"$\Delta t$ (ms)", labelpad=6)
    ax.set_zlabel("activation time (ms)", labelpad=8)
    ax.set_xticks(dx_index, [f"{size:g}" for size in MESH_SIZES])
    ax.set_yticks(dt_index, [f"{dt:g}" for dt in TIME_STEPS])
    ax.set(zlim=(0, 150), zticks=(0, 50, 100, 150),
           xlim=(-0.2, len(MESH_SIZES) - 0.8), ylim=(-0.2, len(TIME_STEPS) - 0.8))
    ax.tick_params(labelsize=7, pad=2, width=0.6, length=2.5)
    ax.view_init(elev=22, azim=40)
    ax.set_box_aspect((1.35, 1.55, 0.85))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("0.75")
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_zticklabels()):
        label.set_clip_on(False)
    save_figure(fig, output_dir / "benchmark_surface.pdf", builds)


def draw_slice(mesh, size, output_dir, build):
    # Figure 1b plane through P1, P8, and the two mid-edge points.
    plane = vtkPlane()
    plane.SetOrigin(P1)
    plane.SetNormal(*np.cross((0, 7, 1.5), (20, 0, 1.5)))
    cutter = vtkCutter()
    cutter.SetInputData(mesh)
    cutter.SetCutFunction(plane)
    cutter.GenerateTrianglesOn()
    cutter.Update()
    cut = cutter.GetOutput()
    if not cut.GetNumberOfPolys():
        raise ValueError("The isochrone plane did not intersect the activation mesh")
    points = vtk_to_numpy(cut.GetPoints().GetData())
    faces = vtk_to_numpy(cut.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    values = vtk_to_numpy(cut.GetPointData().GetArray("ActivationTime"))

    levels = np.arange(0, 131, 10)
    colors = plt.get_cmap("jet")(np.linspace(0, 1, len(levels) + 1))[1:-1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=figure_size(9.2, 3.3), layout="constrained")
    args = (points[:, 0], points[:, 1], faces, values)
    collection = ax.tricontourf(*args, levels=levels, cmap=cmap, norm=norm)
    ax.tricontour(*args, levels=levels, colors="black", linewidths=1.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=(0, 20), ylim=(0, 7), xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.6)
    colorbar = fig.colorbar(collection, ax=ax, ticks=(0, 40, 80, 120), fraction=0.046, pad=0.02)
    colorbar.set_label("activation time (ms)")
    colorbar.ax.tick_params(width=0.6, length=2.5)
    colorbar.outline.set_linewidth(0.6)
    save_figure(fig, output_dir / f"benchmark_slice_dx{size:g}.pdf", {(size, FIGURE_DT): build})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CASES_DIR.parent / "figures")
    args = parser.parse_args()

    surface = np.full((len(MESH_SIZES), len(TIME_STEPS)), np.nan)
    builds = {}
    for i, size in enumerate(MESH_SIZES):
        for j, dt in enumerate(TIME_STEPS):
            summary = results_dir(size, dt) / "activation.json"
            if summary.is_file():
                record = json.loads(summary.read_text())
                p8 = float(record["p8_ms"])
                if not np.isfinite(p8):
                    raise ValueError(f"Nonfinite P8 activation time in {summary}")
                surface[i, j] = p8
                builds[size, dt] = record.get("solver_build")

    fields = {size: results_dir(size, FIGURE_DT) / "activation.vtu" for size in MESH_SIZES}
    if not any(path.is_file() for path in fields.values()) and not np.isfinite(surface).any():
        raise ValueError(f"No activation results found under {CASES_DIR}. Run the benchmark first.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6})

    curves = {}
    for size, path in fields.items():
        if not path.is_file():
            print(f"Skipping spatial plots for missing {path}", file=sys.stderr)
            continue
        mesh = load_activation(path)
        curves[size] = sample_diagonal(mesh)
        if size in SLICE_SIZES:
            draw_slice(mesh, size, args.output_dir, builds.get((size, FIGURE_DT)))

    if curves:
        draw_diagonal(curves, args.output_dir, builds)
    if np.isfinite(surface).any():
        draw_surface(surface, args.output_dir, builds)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"Niederer figures: {error}")
