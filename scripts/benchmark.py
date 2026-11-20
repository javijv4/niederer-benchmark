"""Measure first activation from the solver's VTU snapshots."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


CASES_DIR = Path(__file__).resolve().parents[1] / "cases"
CASE_PATTERN = r"slab_h0\.[125]_dt0\.(?:005|01|05)"
METRICS = ("p8_ms", "latest_ms")


def load_case(name):
    if not re.fullmatch(CASE_PATTERN, name):
        raise ValueError(f"Unknown case {name}")
    case_dir = CASES_DIR / name
    xml = ET.parse(case_dir / "solver.xml")
    general = xml.find("GeneralSimulationParameters")
    output = general.findtext("Save_results_in_folder").strip()
    if Path(output) != Path("results"):
        raise ValueError(f"Unexpected output directory in {case_dir / 'solver.xml'}: {output}")
    return general, case_dir / output


def snapshots(output_dir, general):
    prefix = general.findtext("Name_prefix_of_saved_VTK_files").strip()
    interval = int(general.findtext("Increment_in_saving_VTK_files"))
    start = int(general.findtext("Start_saving_after_time_step"))
    stop = int(general.findtext("Number_of_time_steps"))
    first = ((start + interval - 1) // interval) * interval
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.vtu")
    files = sorted(
        (int(match[1]), path)
        for path in output_dir.glob(f"{prefix}_*.vtu")
        if (match := pattern.fullmatch(path.name))
    )
    if stop % interval or [step for step, _ in files] != list(range(first, stop + 1, interval)):
        raise ValueError(f"Incomplete or unexpected snapshot sequence in {output_dir}")
    return files


def extract(output_dir, general):
    import numpy as np
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter

    dt = float(general.findtext("Time_step_size"))
    files = snapshots(output_dir, general)
    reader = vtkXMLUnstructuredGridReader()

    def read(step, path):
        reader.SetFileName(str(path))
        reader.Update()
        grid = reader.GetOutput()
        values = grid.GetPointData().GetArray("Membrane_potential")
        clock = grid.GetFieldData().GetArray("TimeValue")
        if reader.GetErrorCode() or not grid.GetNumberOfPoints() or values is None:
            raise ValueError(f"Missing or unreadable Membrane_potential in {path}")
        if clock is not None and not math.isclose(
            clock.GetTuple1(0), step * dt, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError(f"Unexpected TimeValue in {path}")
        voltage = np.array(vtk_to_numpy(values), dtype=float, copy=True)
        if voltage.shape != (grid.GetNumberOfPoints(),) or not np.isfinite(voltage).all():
            raise ValueError(f"Invalid Membrane_potential values in {path}")
        return vtk_to_numpy(grid.GetPoints().GetData()), voltage

    first_step, first_path = files[0]
    points, previous = read(first_step, first_path)
    points = points.copy()
    mesh = vtkUnstructuredGrid()
    mesh.CopyStructure(reader.GetOutput())
    p8 = np.flatnonzero(np.all(np.abs(points - (20, 7, 3)) <= 1e-6, axis=1))
    if p8.size != 1:
        raise ValueError(f"Expected one node at P8 (20, 7, 3) mm in {first_path}")
    if np.any(previous >= 0):
        raise ValueError(f"Activation precedes the first saved snapshot in {first_path}")

    activation = np.full(previous.size, np.nan)
    previous_step = first_step
    for step, path in files[1:]:
        current_points, current = read(step, path)
        if not np.array_equal(current_points, points):
            raise ValueError(f"Point ordering or coordinates changed in {path}")
        crossed = np.isnan(activation) & (previous < 0) & (current >= 0)
        fraction = -previous[crossed] / (current[crossed] - previous[crossed])
        activation[crossed] = (previous_step + fraction * (step - previous_step)) * dt
        if np.isfinite(activation).all():
            if step != files[-1][0]:
                read(*files[-1])
            break
        previous, previous_step = current, step

    missing = np.count_nonzero(~np.isfinite(activation))
    if missing:
        raise ValueError(f"{missing} nodes never reached 0 mV in {output_dir}")

    times = numpy_to_vtk(activation, deep=True)
    times.SetName("ActivationTime")
    mesh.GetPointData().SetScalars(times)
    path = output_dir / "activation.vtu"
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(str(path))
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    if writer.Write() != 1:
        raise OSError(f"Could not write {path}")

    metrics = {"p8_ms": float(activation[p8[0]]), "latest_ms": float(activation.max())}
    build = json.loads(os.environ.get("NIEDERER_SOLVER_BUILD", "null"))
    if build is not None:
        metrics["solver_build"] = build
    (output_dir / "activation.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "evaluate", "cleanup"))
    parser.add_argument("case", metavar="CASE")
    parser.add_argument("metric", choices=METRICS, nargs="?")
    args = parser.parse_args()
    if (args.action == "evaluate") != (args.metric is not None):
        parser.error("evaluate requires a metric")
    general, output_dir = load_case(args.case)
    metrics_path = output_dir / "activation.json"
    if args.action == "extract":
        extract(output_dir, general)
    elif args.action == "evaluate":
        value = float(json.loads(metrics_path.read_text())[args.metric])
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite metric {args.metric} in {metrics_path}")
        print(value)
    else:
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        for _, path in snapshots(output_dir, general):
            path.unlink()


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError) as error:
        sys.exit(f"Niederer benchmark: {error}")
