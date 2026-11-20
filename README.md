# Niederer electrophysiology benchmark

The benchmark simulates propagation through a 20 × 7 × 3 mm slab using the
cardiac monodomain equations and the ten Tusscher–Panfilov epicardial model.
Fibres run along the 20 mm axis. A 2 ms stimulus is applied in the
1.5 mm cube at the origin, and the outer boundary has zero flux. The nine
benchmarks combine tetrahedral meshes with nominal spacings of
0.5, 0.2 and 0.1 mm and time steps of 0.05, 0.01 and 0.005 ms. Each benchmark
runs one solve to 50 ms.
[Niederer et al. (2011), §§3–6](https://doi.org/10.1098/rsta.2011.0139)

## Evaluation

Nodal activation times are calculated as the first time the action potential
is at or above 0 mV, interpolated linearly between membrane-potential snapshots
saved every 0.05 ms. Two things are evaluated:

- **Published comparison.** For the finest discretization (0.1 mm, 0.005 ms), the
  benchmark checks that the latest nodal activation is within the published
  37.8–48.7 ms interval for the finest resolution.
- **P8 regression.** Each case embeds the activation time at
  P8, `(20, 7, 3) mm`, from an archived svMultiPhysics (tag `v1.0.0.2`) run.
  The harness compares the new activation time with that value using an
  absolute tolerance of 0.10 ms.

## Running the benchmark

Run `./allRun` from the repository root to run all nine cases. Use either
`--mesh` or `--case` to run a subset.

| Option | Description |
| --- | --- |
| `--solver PATH` | svMultiPhysics executable. Defaults to `SVMP_EXECUTABLE` or `svmultiphysics` on PATH. |
| `--ranks N` | Number of MPI ranks. Defaults to 1. |
| `--mesh {0.5,0.2,0.1}` | Run all three time steps for one mesh size in mm. |
| `--case NAME` | Run a named case, such as `slab_h0.5_dt0.05`. Repeat to select multiple cases. |
| `--figures` | Generate PDFs after the selected cases finish. Off by default. |
| `-h`, `--help` | Show command-line help. |


## Runtime reference

Solver time measured with 12 MPI ranks on one node of an Intel Core Ultra 7 265K,
expressed in core hours (rank count times elapsed solver time). Mesh generation
and activation extraction add a few minutes per mesh.

| Case | Core hours |
| --- | --- |
| `slab_h0.5_dt0.05` | 0.2 |
| `slab_h0.5_dt0.01` | 0.2 |
| `slab_h0.5_dt0.005` | 0.3 |
| `slab_h0.2_dt0.05` | 3.5 |
| `slab_h0.2_dt0.01` | 4.2 |
| `slab_h0.2_dt0.005` | 5.1 |
| `slab_h0.1_dt0.05` | 29 |
| `slab_h0.1_dt0.01` | 41 |
| `slab_h0.1_dt0.005` | 55 |

All nine cases take about 140 core hours. The finest mesh dominates.