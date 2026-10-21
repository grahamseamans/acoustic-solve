"""acoustic_solve.room.field and acoustic_solve.reduce_nodes -- the field anywhere, checked without a solve, on a real
gmsh room mesh (tests/test_room_build.py's two-box room with its field plane: seconds):

- P2 interpolation reproduces a quadratic field exactly at random points of the air, on gmsh's node order and on a
  result's own (nodes shuffled, the edge nodes in VTK's order: the edge pairs are read off the mesh);
- a point outside the air is refused;
- a fake result of that mesh read on its field plane, then REDUCED to the builder's kept nodes (reduce_nodes, as the
  pod runs it): the reduced result gives the same plane, the same seats and the same face averages, is marked, and a
  second reduction leaves it alone; a kept coordinate that is no node is refused.
In the blab venv (gmsh, meshio, numpy, scipy):

    blabrun tests/test_room_field.py
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import meshio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_boundary_lab_results import write_room_result  # noqa: E402
from test_room_build import TWO_BOX_ROOM  # noqa: E402
from test_room_traps import quadratic  # noqa: E402
from acoustic_solve import reduce_nodes  # noqa: E402
from acoustic_solve.boundary_lab_result import BoundaryLabResult  # noqa: E402
from acoustic_solve.room import build, field, spec, traps  # noqa: E402

BUILT = {}
FREQUENCIES_HZ = np.array([30.0, 31.0])
PORTS = ["excitation:front", "excitation:rear", "excitation:side", "excitation:b-front", "excitation:trap:middle",
         "excitation:trap:edge"]
GMSH_TO_VTK_TETRA10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]


def built_stage():
    if "stage" not in BUILT:
        root = Path(tempfile.mkdtemp(prefix="room-field-"))
        (root / "room.json").write_text(json.dumps(TWO_BOX_ROOM))
        BUILT["stage"] = build.main(root / "stage", root / "room.json")
    return BUILT["stage"]


def air_points(count, seed=3):
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < count:
        point = rng.uniform([0, 0, 0], TWO_BOX_ROOM["dimensions_m"])
        if spec.where_not_in_air(TWO_BOX_ROOM, point) is None:
            points.append(point)
    return np.array(points)


def test_p2_interpolation_is_exact_on_a_quadratic_in_either_node_order_and_refuses_a_point_outside():
    room_mesh = meshio.read(built_stage() / "build" / spec.MESH_FILE)
    points = air_points(200)
    gmsh_order = field.P2Mesh.from_arrays(room_mesh.points, room_mesh.cells_dict["tetra10"])
    nodes, weights = gmsh_order.interpolation(points)
    assert np.allclose((weights * quadratic(room_mesh.points)[nodes]).sum(axis=1), quadratic(points), rtol=1e-11)
    shuffle = np.random.default_rng(5).permutation(len(room_mesh.points))
    position_of = np.argsort(shuffle)                       # old node index -> its index in the shuffled order
    other = field.P2Mesh.from_arrays(room_mesh.points[shuffle],
                                     position_of[room_mesh.cells_dict["tetra10"][:, GMSH_TO_VTK_TETRA10]])
    assert other.edge_pairs != gmsh_order.edge_pairs
    nodes, weights = other.interpolation(points)
    assert np.allclose((weights * quadratic(other.nodes)[nodes]).sum(axis=1), quadratic(points), rtol=1e-11)
    try:
        gmsh_order.interpolation([[1.0, 1.0, 2.5]])        # above the 1.8 m ceiling
    except RuntimeError as error:
        assert "in no tetrahedron" in str(error), error
    else:
        raise AssertionError("a point outside the room was interpolated")


def fake_solved(stage):
    """A fake full result of the stage's own mesh: port k's field is (k + 1) x the quadratic, x 1j at the second
    frequency."""
    room_mesh = meshio.read(stage / "build" / spec.MESH_FILE)
    scale = np.array([[k + 1.0 for k in range(len(PORTS))], [1j * (k + 1) for k in range(len(PORTS))]])
    pressure = scale[:, :, None] * quadratic(room_mesh.points)[None, None, :]
    write_room_result(stage / "raw", FREQUENCIES_HZ, PORTS, room_mesh.points, pressure,
                      tetrahedra10=room_mesh.cells_dict["tetra10"])
    return scale


def test_a_plane_reads_the_same_before_and_after_the_pod_keeps_only_the_kept_nodes():
    stage = built_stage()
    scale = fake_solved(stage)
    full_plane, grid = field.plane_pressure(stage, "ear")
    inside = grid["in_air"]
    assert inside.sum() < inside.size, "the plane at 1 m passes through no box: the in_air mask is not exercised"
    expected = scale[:, :, None] * quadratic(grid["points_m"][inside])[None, None, :]
    assert np.allclose(full_plane[:, :, inside], expected, rtol=1e-11) and np.isnan(full_plane[:, :, ~inside]).all()
    full_traps = traps.load_trap_room(stage)
    before = BoundaryLabResult(stage / "raw").domain_coordinate("fem:room", "points_m").shape[0]
    record = reduce_nodes.reduce(str(stage / "raw"), str(stage / "build" / spec.KEPT_NODES_FILE))
    assert record["of_nodes"] == before and record["kept_nodes"] < before / 2, record
    assert reduce_nodes.reduce(str(stage / "raw"), str(stage / "build" / spec.KEPT_NODES_FILE)) == record
    reduced = BoundaryLabResult(stage / "raw")
    assert reduced.domain_coordinate("fem:room", "points_m").shape[0] == record["kept_nodes"]
    assert reduced.domain_topology("fem:room", "tetrahedra10").max() < record["kept_nodes"]
    reduced_plane, _ = field.plane_pressure(stage, "ear")
    assert np.allclose(reduced_plane[:, :, inside], full_plane[:, :, inside], rtol=1e-12)
    reduced_traps = traps.load_trap_room(stage)
    assert np.allclose(reduced_traps.face_pressure, full_traps.face_pressure, rtol=1e-12)
    assert np.allclose(reduced_traps.room.pressure, full_traps.room.pressure, rtol=1e-12)
    elsewhere = Path(tempfile.mkdtemp(prefix="room-field-wrong-")) / "raw"
    shutil.copytree(stage / "raw", elsewhere)
    manifest = json.loads((elsewhere / "manifest.json").read_text())
    del manifest["reduced_to_kept_nodes"]
    (elsewhere / "manifest.json").write_text(json.dumps(manifest))
    np.save(elsewhere.parent / "wrong.npy", np.array([[9.0, 9.0, 9.0]]))
    try:
        reduce_nodes.reduce(str(elsewhere), str(elsewhere.parent / "wrong.npy"))
    except SystemExit as stop:
        assert "no node of the result" in str(stop), stop
    else:
        raise AssertionError("a kept coordinate off every node was accepted")


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
