"""acoustic_solve.room.build -- the room builder without a solve: the spec's seat points and box frame, a port that
does not fit its face refused, and ONE SMALL ROOM BUILT (gmsh, a 2.4 x 2 x 1.8 m room at a coarse mesh: seconds) --
its groups found (every port one surface, the walls, a trap's absorbing face), every seat point a mesh node, the
project's boundaries and loss, the stage's files -- then submitted through acoustic_solve.solve_job (JOB written, no
pod). The free field's box mesh is closed and outward. In the blab venv (gmsh, podd):

    blabrun tests/test_room_build.py
"""
import json
import math
import tempfile
import time
from pathlib import Path

import meshio
import numpy as np
from podd import job_contract

from acoustic_solve import pins
from acoustic_solve.room import build, spec

ROOM = {"name": "tiny-live-test", "dimensions_m": [2.4, 2.0, 1.8],
        "source": {"size_m": [0.3, 0.3, 0.3], "centre_m": [0.5, 1.0, 0.6], "aim_m": [2.0, 1.0],
                   "ports": [{"name": "front", "face": "front", "shape": "face"},
                             {"name": "rear", "face": "rear", "shape": "face"},
                             {"name": "side", "face": "left", "shape": "disc", "at_m": [0.0, 0.0], "radius_m": 0.05}],
                   "free_field_points_m": {"axis-1m": [1.15, 0.0, 0.0]}},
        "seats_m": {"centre": [1.8, 1.0, 1.0]}, "seat_grid": {"step_m": 0.1, "half_count": 1},
        "h_far_m": 0.45, "h_near_m": 0.12, "absorption": {"bulk_loss_factor": 0.05, "rho_c_patch_fraction": 0.0},
        "frequencies_hz": [40.0, 41.0], "traps": [{"corner": [1, 1], "width_m": 0.5, "height_m": 1.0}]}


def test_seat_points_are_a_grid_at_ear_height_and_the_centre_lower_and_higher():
    points = spec.seat_points(ROOM)
    assert len(points) == 9 + 2 and np.allclose(points["centre/x+1y-1"], [1.9, 0.9, 1.0])
    assert np.allclose(points["centre/z-0.2"], [1.8, 1.0, 0.8]) and np.allclose(points["centre/z+0.2"], [1.8, 1.0, 1.2])
    centre, grid = spec.seat_point_indices(list(points), "centre")
    assert list(points)[centre] == "centre/x+0y+0" and len(grid) == 9


def test_the_box_frame_puts_the_aim_on_plus_x():
    source = {**ROOM["source"], "centre_m": [1.0, 1.0, 0.5], "aim_m": [1.0, 2.0]}
    assert np.allclose(spec.in_box_frame(source, [[1.0, 1.7, 0.5]]), [[0.7, 0.0, 0.0]])


def test_a_port_that_does_not_fit_its_face_is_refused():
    port = {"name": "big", "face": "front", "shape": "disc", "at_m": [0.1, 0.0], "radius_m": 0.1}
    try:
        build.port_geometry(ROOM["source"], port)
    except RuntimeError as error:
        assert "does not fit" in str(error)
    else:
        raise AssertionError("a disc past its face's edge was accepted")


def test_a_small_room_builds_with_every_seat_a_node_and_submits_through_solve_job():
    root = Path(tempfile.mkdtemp(prefix="room-build-"))
    (root / "room.json").write_text(json.dumps(ROOM))
    stage = build.main(root / "stage", root / "room.json")
    facts = json.loads((stage / spec.POINTS_FILE).read_text())
    assert facts["area_m2"]["front"] > 0 and abs(facts["area_m2"]["side"] - math.pi * 0.05 ** 2) < 1e-3
    assert facts["area_m2"]["traps"] > 0 and facts["element_types"] == [11], facts["element_types"]   # P2 tetrahedra
    nodes = meshio.read(stage / "build" / build.MESH_FILE).points
    for name, point in facts["seat_points_m"].items():
        assert np.linalg.norm(nodes - point, axis=1).min() < 1e-9, f"{name} is not a node"
    project = json.loads((stage / "build" / build.PROJECT_FILE).read_text())["physical_system"]
    kinds = {boundary["name"]: boundary["kind"] for boundary in project["boundaries"]}
    assert kinds == {"front": "moving", "rear": "moving", "side": "moving", "box": "rigid", "walls": "rigid",
                     "traps": "plane_wave_tube_termination"}, kinds
    assert project["regions"][0]["loss_model"] == {"bulk_loss_factor": 0.05}
    assert json.loads((stage / build.REQUEST_FILE).read_text())["retain"] == ["fem_nodal_pressure"]
    job = build.submit(stage, expected_s=120, ram_gb=2, cpu=1.5)
    assert job_contract.stage_state(stage) == job_contract.JOB and job["requires"]["image"] == pins.POD_IMAGE
    assert job["command"] == f"bash run-job.sh build/{build.PROJECT_FILE} {build.REQUEST_FILE} beat_cpu 2"
    assert job["resumable"] is True and job["requires"]["gpu"] == "none"


def test_the_free_field_box_is_closed_outward_with_its_ports_and_probes_per_spec():
    root = Path(tempfile.mkdtemp(prefix="free-field-build-"))
    (root / "a.json").write_text(json.dumps(ROOM))
    (root / "b.json").write_text(json.dumps({**ROOM, "name": "tiny-moved", "source": {**ROOM["source"],
                                                                                      "centre_m": [1.0, 0.8, 0.6]}}))
    stage = build.free_field(root / "free", [root / "a.json", root / "b.json"])
    probes = json.loads((stage / spec.FREE_FIELD_SPECS_FILE).read_text())["probes"]
    assert set(probes) == {"tiny-live-test", "tiny-moved", spec.FREE_FIELD_BOX_PROBE}
    request = json.loads((stage / build.REQUEST_FILE).read_text())
    assert [probe["id"] for probe in request["probes"]] == ["tiny-live-test", "tiny-moved", "box"]
    (root / "c.json").write_text(json.dumps({**ROOM, "name": "other-box",
                                             "source": {**ROOM["source"], "size_m": [0.4, 0.3, 0.3]}}))
    try:
        build.free_field(root / "free2", [root / "a.json", root / "c.json"])
    except RuntimeError as error:
        assert "another box" in str(error)
    else:
        raise AssertionError("two boxes in one free field")


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
