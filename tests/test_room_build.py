"""acoustic_solve.room.build -- the room builder without a solve: the spec's seat points and box frame, a port that
does not fit its face refused, and ONE SMALL ROOM BUILT (gmsh, a 2.4 x 2 x 1.8 m room at a coarse mesh: seconds) --
its groups found (every port one surface, the walls, each trap's face "trap:<name>"), every seat and trap mic point a
mesh node, the project's boundaries and loss (a velocity_port trap a moving boundary + component + port, a rho_c one
the termination), the stage's files -- then submitted through acoustic_solve.solve_job (JOB written, no pod). A trap
without its face kind, and a mic point outside the air, are refused. The free field's box mesh is closed and
outward. In the blab venv (gmsh, podd):

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
        "sources": [{"name": "speaker", "size_m": [0.3, 0.3, 0.3], "centre_m": [0.5, 1.0, 0.6], "aim_m": [2.0, 1.0],
                     "ports": [{"name": "front", "face": "front", "shape": "face"},
                               {"name": "rear", "face": "rear", "shape": "face"},
                               {"name": "side", "face": "left", "shape": "disc", "at_m": [0.0, 0.0],
                                "radius_m": 0.05}],
                     "free_field_points_m": {"axis-1m": [1.15, 0.0, 0.0]}}],
        "seats": {"centre": {"centre_m": [1.8, 1.0, 1.0], "step_m": [0.1, 0.1, 0.2], "half_count": [1, 1, 1]}},
        "h_far_m": 0.45, "h_near_m": 0.12, "absorption": {"bulk_loss_factor": 0.05, "rho_c_patch_fraction": 0.0},
        "frequencies_hz": [40.0, 41.0],
        "traps": [{"name": "far", "face": "velocity_port", "mount": "corner", "corner": [1, 1], "level": "floor",
                   "width_m": 0.5, "height_m": 1.0,
                   "mic_points_m": {"front": [0.1, 0.0, 0.0], "high": [0.05, 0.1, 0.3]}},
                  {"name": "near", "face": "rho_c", "mount": "corner", "corner": [0, 0], "level": "floor",
                   "width_m": 0.4, "height_m": 0.8},
                  {"name": "top", "face": "velocity_port", "mount": "corner", "corner": [1, 0], "level": "ceiling",
                   "width_m": 0.4, "height_m": 0.5},
                  {"name": "beside", "face": "velocity_port", "mount": "wall", "wall": "y0", "centre_m": [1.5, 0.0, 1.0],
                   "width_m": 0.4, "height_m": 0.3, "mic_points_m": {"front": [0.1, 0.0, 0.0]}},
                  {"name": "roof", "face": "velocity_port", "mount": "wall", "wall": "z1", "centre_m": [1.2, 1.2, 1.8],
                   "width_m": 0.3, "height_m": 0.2}]}


def test_seat_points_are_a_named_box_of_points_and_its_ear_height_grid():
    points = spec.seat_points(ROOM)
    assert len(points) == 27 and np.allclose(points["centre/x+1y-1z+0"], [1.9, 0.9, 1.0])
    assert np.allclose(points["centre/x+0y+0z-1"], [1.8, 1.0, 0.8])
    centre, grid = spec.seat_point_indices(list(points), "centre")
    assert list(points)[centre] == "centre/x+0y+0z+0" and len(grid) == 9


def test_the_box_frame_puts_the_aim_on_plus_x():
    box = {**ROOM["sources"][0], "centre_m": [1.0, 1.0, 0.5], "aim_m": [1.0, 2.0]}
    assert np.allclose(spec.in_box_frame(box, [[1.0, 1.7, 0.5]]), [[0.7, 0.0, 0.0]])


def test_a_port_that_does_not_fit_its_face_is_refused():
    port = {"name": "big", "face": "front", "shape": "disc", "at_m": [0.1, 0.0], "radius_m": 0.1}
    try:
        build.port_geometry(ROOM["sources"][0], port)
    except RuntimeError as error:
        assert "does not fit" in str(error)
    else:
        raise AssertionError("a disc past its face's edge was accepted")


def refused(spec_to_build, words):
    root = Path(tempfile.mkdtemp(prefix="room-refused-"))
    (root / "room.json").write_text(json.dumps(spec_to_build))
    try:
        build.main(root / "stage", root / "room.json")
    except RuntimeError as error:
        assert words in str(error), error
    else:
        raise AssertionError(f"built a spec that should be refused for {words!r}")


def test_a_trap_without_its_face_kind_mount_or_name_is_refused_and_wall_faces_must_fit_and_not_overlap():
    far, near, top, beside, roof = ROOM["traps"]
    rest = [near, top, beside, roof]
    refused({**ROOM, "traps": [{k: v for k, v in far.items() if k != "face"}, *rest]}, "required, no default")
    refused({**ROOM, "traps": [{**far, "face": "absorber"}, *rest]}, "is not one of")
    refused({**ROOM, "traps": [{k: v for k, v in far.items() if k != "mount"}, *rest]}, "mount None")
    refused({**ROOM, "traps": [{k: v for k, v in far.items() if k != "level"}, *rest]}, "needs")
    refused({**ROOM, "traps": [{k: v for k, v in far.items() if k != "name"}, *rest]}, "name None")
    refused({**ROOM, "traps": [far, near, top, {**beside, "name": "far"}, roof]}, "not unique")
    refused({**ROOM, "traps": [{**far, "mic_points_m": {"face": [0.1, 0, 0]}}, *rest]}, "mic name")
    refused({**ROOM, "traps": [far, near, top, {**beside, "centre_m": [1.5, 0.1, 1.0]}, roof]}, "is not on surface")
    refused({**ROOM, "traps": [far, near, top, {**beside, "centre_m": [2.3, 0.0, 1.0]}, roof]}, "past its surface")
    refused({**ROOM, "traps": [*ROOM["traps"], {**beside, "name": "twin", "centre_m": [1.7, 0.0, 1.1]}]}, "overlap")
    refused({**ROOM, "sources": [ROOM["sources"][0], {**ROOM["sources"][0], "name": "other"}]}, "not unique")
    refused({k: v for k, v in ROOM.items() if k != "sources"}, "non-empty list")


def test_a_mic_point_outside_the_air_is_refused():
    far, *rest = ROOM["traps"]
    for mic, where in (([-0.05, 0.0, 0.0], "prism"), ([0.0, 0.0, 0.0], "prism"),     # behind / on the face
                       ([0.1, 0.0, 1.4], "walls"),                                    # above the 1.8 m ceiling
                       ([-0.8, 0.0, 0.0], "walls"),                                   # through the corner behind
                       ([1.8, -0.636, 0.1], "source box")):                           # the box's centre (0.5, 1, 0.6)
        refused({**ROOM, "traps": [{**far, "mic_points_m": {"m": mic}}, *rest]}, where)


def test_a_small_room_builds_with_every_seat_a_node_and_submits_through_solve_job():
    root = Path(tempfile.mkdtemp(prefix="room-build-"))
    (root / "room.json").write_text(json.dumps(ROOM))
    stage = build.main(root / "stage", root / "room.json")
    facts = json.loads((stage / spec.POINTS_FILE).read_text())
    assert facts["area_m2"]["front"] > 0 and abs(facts["area_m2"]["side"] - math.pi * 0.05 ** 2) < 1e-3
    for trap, area in (("far", 0.5), ("near", 0.32), ("top", 0.2), ("beside", 0.12), ("roof", 0.06)):
        assert abs(facts["area_m2"][f"trap:{trap}"] - area) < 1e-9, (trap, facts["area_m2"])
    assert facts["element_types"] == [11], facts["element_types"]                                    # P2 tetrahedra
    assert list(facts["trap_points_m"]) == ["far/front", "far/high", "beside/front"]
    # the far trap's face centre (2.4 - l/2, 2.0 - l/2, 0.5), l = 0.5 / sqrt 2; normal into the room (-1, -1) / sqrt 2
    leg = 0.5 / math.sqrt(2)
    assert np.allclose(facts["trap_points_m"]["far/front"], [2.4 - leg / 2 - 0.1 / math.sqrt(2),
                                                             2.0 - leg / 2 - 0.1 / math.sqrt(2), 0.5])
    assert np.allclose(facts["trap_points_m"]["beside/front"], [1.5, 0.1, 1.0])     # y0 wall: normal +y
    assert "kept_nodes" not in facts and not (stage / "build" / spec.KEPT_NODES_FILE).exists()
    nodes = meshio.read(stage / "build" / spec.MESH_FILE).points
    for name, point in {**facts["seat_points_m"], **facts["trap_points_m"]}.items():
        assert np.linalg.norm(nodes - point, axis=1).min() < 1e-9, f"{name} is not a node"
    project = json.loads((stage / "build" / build.PROJECT_FILE).read_text())["physical_system"]
    kinds = {boundary["name"]: boundary["kind"] for boundary in project["boundaries"]}
    assert kinds == {"front": "moving", "rear": "moving", "side": "moving", "box": "rigid", "walls": "rigid",
                     "trap:far": "moving", "trap:near": "plane_wave_tube_termination", "trap:top": "moving",
                     "trap:beside": "moving", "trap:roof": "moving"}, kinds
    assert [port["id"] for port in project["excitation_ports"]] == [
        "excitation:front", "excitation:rear", "excitation:side", "excitation:trap:far", "excitation:trap:top",
        "excitation:trap:beside", "excitation:trap:roof"]
    trap_component = next(c for c in project["components"] if c["id"] == "component:trap:far")
    assert trap_component["kind"] == "ideal_velocity_source"
    assert trap_component["boundary_ids"] == ["boundary:trap:far"]
    assert project["regions"][0]["loss_model"] == {"bulk_loss_factor": 0.05}
    assert json.loads((stage / build.REQUEST_FILE).read_text())["retain"] == ["fem_nodal_pressure"]
    job = build.submit(stage, expected_s=120, ram_gb=2, cpu=1.5)
    assert job_contract.stage_state(stage) == job_contract.JOB and job["requires"]["image"] == pins.POD_IMAGE
    assert job["command"] == f"bash run-job.sh build/{build.PROJECT_FILE} {build.REQUEST_FILE} beat_cpu 2"
    assert job["resumable"] is True and job["requires"]["gpu"] == "none"


TWO_BOX_ROOM = {**ROOM, "name": "two-boxes-patched",
                "sources": [ROOM["sources"][0], {**ROOM["sources"][0], "name": "speaker-b", "centre_m": [0.5, 0.4, 0.6],
                                                 "aim_m": [2.0, 0.4],
                                                 "ports": [{"name": "b-front", "face": "front", "shape": "face"}]}],
                "absorption": {"bulk_loss_factor": 0.01, "rho_c_patch_fraction": 0.25},
                "traps": [{"name": "middle", "face": "velocity_port", "mount": "wall", "wall": "x1",
                           "centre_m": [2.4, 1.0, 0.9], "width_m": 0.4, "height_m": 0.4},
                          {"name": "edge", "face": "velocity_port", "mount": "wall", "wall": "y1",
                           "centre_m": [1.2 + 0.6, 2.0, 0.9], "width_m": 0.4, "height_m": 0.4}],
                "field_planes": [{"name": "ear", "normal_axis": "z", "at_m": 0.6, "step_m": 0.2}]}


def test_two_boxes_and_wall_traps_cut_out_of_the_patches_and_a_field_plane_keeps_only_what_is_read():
    root = Path(tempfile.mkdtemp(prefix="room-two-boxes-"))
    (root / "room.json").write_text(json.dumps(TWO_BOX_ROOM))
    stage = build.main(root / "stage", root / "room.json")
    facts = json.loads((stage / spec.POINTS_FILE).read_text())
    area = facts["area_m2"]
    assert abs(area["trap:middle"] - 0.16) < 1e-9 and abs(area["trap:edge"] - 0.16) < 1e-9, area
    walls_total = 2 * (2.4 * 2.0 + 2.4 * 1.8 + 2.0 * 1.8)
    # the patches absorb 0.25 of every surface: grown around the trap faces cut out of them (middle wholly, edge partly)
    assert abs(area["patches"] - 0.25 * walls_total) < 1e-6, area["patches"]
    assert area["b-front"] > 0 and abs(area["box"] - (2 * 6 * 0.09 - 3 * 0.09 - math.pi * 0.05 ** 2)) < 2e-3, area
    assert set(facts["yaw_deg"]) == {"speaker", "speaker-b"}
    project = json.loads((stage / "build" / build.PROJECT_FILE).read_text())["physical_system"]
    kinds = {boundary["name"]: boundary["kind"] for boundary in project["boundaries"]}
    assert kinds["patches"] == "plane_wave_tube_termination" and kinds["trap:middle"] == "moving", kinds
    kept = np.load(stage / "build" / spec.KEPT_NODES_FILE)
    assert (stage / "build" / "reduce_nodes.py").is_file() and facts["kept_nodes"] == len(kept) < facts["nodes"]
    room_mesh = meshio.read(stage / "build" / spec.MESH_FILE)
    tree = {tuple(np.round(p, 9)) for p in kept}
    for name, point in facts["seat_points_m"].items():
        assert tuple(np.round(point, 9)) in tree, f"seat point {name} is not kept"
    for group in ("front", "b-front", "trap:middle", "trap:edge"):
        triangles = room_mesh.cells_dict["triangle6"][room_mesh.cell_sets_dict[group]["triangle6"]]
        assert all(tuple(np.round(p, 9)) in tree for p in room_mesh.points[np.unique(triangles)]), group


def test_the_free_field_boxes_are_closed_outward_with_their_ports_and_probes_per_spec():
    root = Path(tempfile.mkdtemp(prefix="free-field-build-"))
    (root / "a.json").write_text(json.dumps(TWO_BOX_ROOM))
    moved = [{**box, "centre_m": [box["centre_m"][0] + 0.5, box["centre_m"][1] + 0.2, 0.6],
              "aim_m": [box["aim_m"][0] + 0.5, box["aim_m"][1] + 0.2]} for box in TWO_BOX_ROOM["sources"]]
    (root / "b.json").write_text(json.dumps({**TWO_BOX_ROOM, "name": "tiny-moved", "sources": moved}))
    stage = build.free_field(root / "free", [root / "a.json", root / "b.json"])
    probes = json.loads((stage / spec.FREE_FIELD_SPECS_FILE).read_text())["probes"]
    assert set(probes) == {"two-boxes-patched", "tiny-moved", spec.FREE_FIELD_BOX_PROBE}
    assert probes[spec.FREE_FIELD_BOX_PROBE] == ["speaker/axis-1m", "speaker-b/axis-1m"]
    request = json.loads((stage / build.REQUEST_FILE).read_text())
    assert [probe["id"] for probe in request["probes"]] == ["two-boxes-patched", "tiny-moved", "box"]
    # the second box's axis point, 1.15 m in front of it, in the first box's frame (both face +x, b 0.6 m to the right)
    assert np.allclose(request["probes"][2]["points_m"][1], [1.15, -0.6, 0.0])
    apart = [{**moved[0]}, {**moved[1], "centre_m": [1.0, 0.5, 0.6]}]
    (root / "c.json").write_text(json.dumps({**TWO_BOX_ROOM, "name": "apart", "sources": apart}))
    try:
        build.free_field(root / "free2", [root / "a.json", root / "c.json"])
    except RuntimeError as error:
        assert "other boxes" in str(error)
    else:
        raise AssertionError("boxes standing otherwise in one free field")


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
