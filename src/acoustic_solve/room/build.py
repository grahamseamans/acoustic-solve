"""A ROOM: a closed shoebox of air with source boxes in it, as a Boundary Lab INTERIOR FEM project, staged as a podd
JOB. Its first studies were the speaker project's (not public: PLANS/2026-10-14-room-cardioid-sub.md,
two pistons on a cube; PLANS/2026-10-15-room-a2-crossover.md, A2's box and its four sources); then the active bass
trap's (PLANS/2026-10-21-rooms-for-the-trap-visuals.md: a stereo pair, corner and wall traps, field planes).

    blabrun -m acoustic_solve.room.build <stage dir> <room.json>          (the Boundary Lab venv: gmsh + blab)
    blabrun -m acoustic_solve.room.build freefield <stage dir> <room.json>...
    blabrun -m acoustic_solve.room.build submit <stage dir> <expected_s> <ram_gb> <cpu>
        (a CPU-only job through acoustic_solve.solve_job: requires.gpu "none", so podd runs it on a CPU pod; <cpu>
        cores, its Julia/BLAS threads; <expected_s>: its time on a FAST host, podd's deadline is 4x that x the pod's
        measured slowness. RESUMABLE per frequency: run-job.sh writes `progress` and resumes the raw/ an earlier,
        stopped attempt brought home, so a chunk stopped on a slow pod or at its deadline is requeued and solves only
        the rest. The stage must be in one of podd's queue directories.)

Why interior FEM and nothing else (cited in monitoring's PLANS/2026-10-14-room-cardioid-sub.md):
  - a system with bounded regions and no unbounded region IS an interior FEM solve
    (~/boundary-lab/src/blab/physical_model.py:160-167, docs/advanced/Interior FEM Solver.md);
  - it is sparse direct on the CPU, fp64, whatever backend is asked (src/blab/solvers/coupled_backend.py:450-465),
    one LU per frequency shared by every port (BEAT BeatEngineCoupledDriver.jl:2370-2392);
  - ten-node tetrahedra are accepted only there (BeatEngineCoupled.jl:1262), so the room is meshed P2;
  - walls: rigid, plane-wave termination (Z = rho c) and Miki on rigid; no `impedance` role
    (coupled_backend.py:687-712); room loss: the region's bulk_loss_factor (Q ~ 1/eta, docs/Coupled Solver.md);
  - point probes are refused for interior FEM (src/blab/headless.py:270-274), so the seat points are EMBEDDED as
    mesh nodes and read out of the retained fem_nodal_pressure; anywhere else the field is interpolated
    (acoustic_solve.room.field).

THE SOURCE BOXES (`sources`, a list): each a rigid rectangular box named `name`, `size_m` [depth, width, height], its
centre at `centre_m`, yawed so its front face (+depth) points at `aim_m` (a horizontal point). Its BOX FRAME: x along
the depth (front face at +depth/2), y across (left face at +width/2), z up. Each of its `ports` is a patch on one
face, its own ideal-velocity component and excitation port `excitation:<name>` (1 m/s uniform normal velocity, along
the FEM domain's outward normal, i.e. INTO the box), so any mix of sources is a post-processing weight
(acoustic_solve.weighted_sums, acoustic_solve.room.analysis). A port: {"name", "face": front | rear | left | right,
"shape": "face" (the whole face) | "disc" ("radius_m") | "rect" ("size_m": [h, v]), "at_m": [h, v]} where (h, v) are
the face's in-plane coordinates of the patch centre: (y, z) on front / rear, (x, z) on left / right. Port names are
unique across the boxes. Everything of every box that is no port is rigid (the group "box").

SEATS (`seats`: {seat: {"centre_m", "step_m": [sx, sy, sz], "half_count": [nx, ny, nz]}}): each a box of points
along the room's axes, named "<seat>/x+1y-1z+0" (the centre "x+0y+0z+0"); every point an embedded mesh node.

TRAPS (optional, `traps`): active bass traps (the active bass trap project), each `mount`ed (required, no default):
  - "corner": a triangular prism in the vertical corner `corner` [i, j] (cut out of the air), standing on the floor
    or hanging from the ceiling (`level`), its 45-degree face `width_m` wide and `height_m` tall, its other ends rigid;
  - "wall": a face `width_m` x `height_m` flush in the room surface `wall` (x0 | x1 | y0 | y1 | z0 | z1, the cabinet
    behind it, outside the air), centred at `centre_m` (on that surface); a wall's width is horizontal, its height
    z; the floor's and ceiling's width is x, height y. No rho c tile touches a trap (its face, or a prism's
    footprint), and the surface's other tiles grow to keep its absorbing area (spec.rho_c_tiles): a trap never takes
    absorption from the room.
Each face is the physical group "trap:<name>" (acoustic_solve.room.spec.trap_group), of one of two kinds (required,
no default; PLANS/2026-10-20-trap-velocity-port.md):
  - "velocity_port": the face is a PORT like the sources' (a moving boundary, an ideal-velocity component, the
    excitation port `excitation:trap:<name>`, 1 m/s uniform normal velocity along the FEM domain's outward normal,
    i.e. INTO the trap). The trap's driver is a rigid piston, so ANY trap behaviour is a law closing the ports in
    post-processing (acoustic_solve.room.traps): one solve serves every law.
  - "rho_c": the face is a plane-wave (rho c) termination like the wall patches, an ideal broadband absorber at
    every frequency; it stays only as the velocity port's equivalence reference.
Its MIC POINTS (optional `mic_points_m`: {mic: [normal, along, up]}) are in its FACE FRAME (spec.trap_face_frame):
origin the face centre, normal into the room; each one an embedded mesh node like the seat points, named
"<trap>/<mic>", written apart from them (seat_points.json's "trap_points_m"). A point (seat or mic) not in the room's
air is refused. The mesh is h_near_m at the trap faces as at the source boxes (growing to h_far_m from 0.05 to 0.6 m
away).

FIELD PLANES (optional `field_planes`: [{"name", "normal_axis", "at_m", "step_m"}], spec.field_plane_points): grids
the field is read on by interpolation (acoustic_solve.room.field). When given, the pod KEEPS ONLY the nodes the
readers need -- the tetrahedra holding the planes' points, the embedded points, every port's and trap's face -- so a
room with many ports comes home small: build/keep_nodes.npy (their coordinates) and build/reduce_nodes.py (run by
run-job.sh after the solve). Without field planes the whole field comes home.

THE FREE FIELD: the room's effect on a source mix is its seat pressure over the SAME boxes' anechoic pressure at the
same points, so one exterior BEM of every box (every port, rigid elsewhere) in the FIRST box's frame is solved with
the seat points of every room spec given, moved into that frame, as point probes, one probe per spec named after it;
plus each box's `free_field_points_m` ({name: [x, y, z]} in ITS box frame), probe "box", named "<box>/<point>". A
rigid yawed translation of the whole set is all a placement is, so one solve serves every spec whose boxes are the same
and stand the same way relative to the first (spec.the_boxes).

<room.json>: {"name", "dimensions_m": [Lx, Ly, Lz], "sources": [{"name", "size_m", "centre_m", "aim_m", "ports",
optional "free_field_points_m"}], "seats", "h_far_m", "h_near_m", "absorption": {"bulk_loss_factor": eta,
"rho_c_patch_fraction": f (0: rigid walls; the absorbing fraction of every surface, its trap faces not counted),
"rho_c_tile_m": the tiles' pitch, with f > 0 (spec.rho_c_tiles: every surface in equal cells, each a centred rho c
rectangle; a pitch past a surface's size is one centred patch)},
"frequencies_hz": [...], optional "traps": [{"name", "face": "velocity_port" | "rho_c", "mount": "corner" | "wall",
"width_m", "height_m", corner: "corner": [i, j] (0: the x = 0 / y = 0 wall, 1: the far one), "level": "floor" |
"ceiling"; wall: "wall", "centre_m"; optional "mic_points_m"}], optional "field_planes"}
"""
import json
import math
import shutil
import sys
from pathlib import Path

import gmsh
import meshio
import numpy as np

from acoustic_solve import solve_job
from acoustic_solve.room.field import P2Mesh
from acoustic_solve.room.spec import (DENSITY_KG_PER_M3, FREE_FIELD_BOX_PROBE, FREE_FIELD_SPECS_FILE,
                                      KEPT_NODES_FILE, MESH_FILE, ON_SURFACE_M, POINTS_FILE, SOUND_SPEED_M_PER_S,
                                      SPEC_FILE, SURFACES, checked_sources, checked_traps, embedded_points,
                                      rho_c_tiles,
                                      field_plane_points,
                                      in_box_frame, seat_points, the_boxes, trap_face_frame, trap_group, trap_points,
                                      wall_face_rectangle, yaw_of)

PROJECT_FILE = "room.blab.json"
FREE_FIELD_MESH_FILE = "box.msh"
FREE_FIELD_PROJECT_FILE = "freefield.blab.json"
FREE_FIELD_ELEMENT_M = 0.025         # lambda / 27 at 500 Hz
REQUEST_FILE = "request.json"
FACES = ("front", "rear", "left", "right")
REDUCER_SCRIPT = Path(__file__).resolve().parents[1] / "reduce_nodes.py"


def port_geometry(box, port):
    """(face normal axis, sign, the patch's centre in the box frame, its in-plane half-extents, its area)."""
    depth, width, height = box["size_m"]
    axis, sign = {"front": (0, 1.0), "rear": (0, -1.0), "left": (1, 1.0), "right": (1, -1.0)}[port["face"]]
    face_h = width if axis == 0 else depth
    if port["shape"] == "face":
        h, v, half_h, half_v, area = 0.0, 0.0, face_h / 2, height / 2, face_h * height
    elif port["shape"] == "disc":
        (h, v), r = port["at_m"], port["radius_m"]
        half_h, half_v, area = r, r, math.pi * r * r
    elif port["shape"] == "rect":
        (h, v), (dh, dv) = port["at_m"], port["size_m"]
        half_h, half_v, area = dh / 2, dv / 2, dh * dv
    else:
        raise RuntimeError(f"(╯°□°)╯ port {port['name']}: shape {port['shape']!r} is not face / disc / rect")
    if abs(h) + half_h > face_h / 2 + 1e-9 or abs(v) + half_v > height / 2 + 1e-9:
        raise RuntimeError(f"(╯°□°)╯ port {port['name']} does not fit its {port['face']} face")
    plane = sign * (depth if axis == 0 else width) / 2
    centre = [plane, h, v] if axis == 0 else [h, plane, v]
    return axis, sign, centre, half_h, half_v, area


def source_shapes(occ, box):
    """A box and its port patches at the box frame's origin -> (box volume tag, [(port, surface tag)])."""
    depth, width, height = box["size_m"]
    volume = occ.addBox(-depth / 2, -width / 2, -height / 2, depth, width, height)
    patches = []
    for port in box["ports"]:
        if port["face"] not in FACES:
            raise RuntimeError(f"(╯°□°)╯ port {port['name']}: face {port['face']!r} is not one of {FACES}")
        axis, sign, (cx, cy, cz), half_h, half_v, _ = port_geometry(box, port)
        normal = [sign, 0.0, 0.0] if axis == 0 else [0.0, sign, 0.0]
        if port["shape"] == "disc":
            surface = occ.addDisk(cx, cy, cz, port["radius_m"], port["radius_m"], zAxis=normal,
                                  xAxis=[0.0, 0.0, 1.0])
        else:
            if axis == 0:
                corners = [(cx, cy - half_h, cz - half_v), (cx, cy + half_h, cz - half_v),
                           (cx, cy + half_h, cz + half_v), (cx, cy - half_h, cz + half_v)]
            else:
                corners = [(cx - half_h, cy, cz - half_v), (cx + half_h, cy, cz - half_v),
                           (cx + half_h, cy, cz + half_v), (cx - half_h, cy, cz + half_v)]
            surface = planar_quad(occ, corners)
        patches.append((port, surface))
    return volume, patches


def planar_quad(occ, corners):
    tags = [occ.addPoint(*corner) for corner in corners]
    lines = [occ.addLine(tags[k], tags[(k + 1) % 4]) for k in range(4)]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def classify_source_surface(box, point_box_frame, area):
    """A surface of one box (its centre of mass in that box's frame, its area) -> the port it is, or "box". A port is
    told by its centre AND its area: a face with a port cut out of it has the face's centre, not the port's area."""
    found = []
    for port in box["ports"]:
        axis, _, centre, half_h, half_v, port_area = port_geometry(box, port)
        h_axis = 1 if axis == 0 else 0
        if (abs(point_box_frame[axis] - centre[axis]) < 1e-6
                and abs(point_box_frame[h_axis] - centre[h_axis]) <= half_h + 1e-6
                and abs(point_box_frame[2] - centre[2]) <= half_v + 1e-6
                and abs(area - port_area) < 2e-3 * port_area):
            found.append(port["name"])
    if len(found) > 1:
        raise RuntimeError(f"(╯°□°)╯ a source surface at {point_box_frame} matches ports {found}")
    return found[0] if found else "box"


def on_box(box, point_box_frame):
    depth, width, height = box["size_m"]
    x, y, z = point_box_frame
    inside = abs(x) <= depth / 2 + 1e-6 and abs(y) <= width / 2 + 1e-6 and abs(z) <= height / 2 + 1e-6
    return inside and min(abs(abs(x) - depth / 2), abs(abs(y) - width / 2), abs(abs(z) - height / 2)) < 1e-6


def surface_of_boxes(boxes, point, area, frame_of):
    """The group a surface of the boxes is (a port name or "box"); frame_of(box, [point]) puts a point in that box's
    frame. A surface on no box, or on two, RAISES."""
    owners = [box for box in boxes if on_box(box, frame_of(box, [point])[0])]
    if len(owners) != 1:
        raise RuntimeError(f"(╯°□°)╯ a surface at {point} is on {len(owners)} source boxes")
    return classify_source_surface(owners[0], frame_of(owners[0], [point])[0], area)


def mesh(spec, path):
    """Mesh the room (gmsh, P2) to path -> the mesh facts (nodes, tetrahedra, group areas, points)."""
    lx, ly, lz = spec["dimensions_m"]
    boxes = checked_sources(spec)
    occ = gmsh.model.occ
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("room")
        room = occ.addBox(0, 0, 0, lx, ly, lz)
        box_volumes, patches = [], []
        for box in boxes:
            volume, box_patches = source_shapes(occ, box)
            moved = [(3, volume)] + [(2, surface) for _, surface in box_patches]
            occ.rotate(moved, 0, 0, 0, 0, 0, 1, yaw_of(box))
            occ.translate(moved, *box["centre_m"])
            box_volumes.append((3, volume))
            patches += box_patches
        traps = checked_traps(spec)
        prisms = {trap["name"]: trap_prism(occ, trap, spec["dimensions_m"]) for trap in traps
                  if trap["mount"] == "corner"}
        wall_faces = {trap["name"]: wall_face(occ, trap, spec["dimensions_m"]) for trap in traps
                      if trap["mount"] == "wall"}
        air, _ = occ.cut([(3, room)], box_volumes + [(3, prism) for prism, _, _ in prisms.values()])
        fraction = spec["absorption"]["rho_c_patch_fraction"]
        surfaces_in = [(2, surface) for _, surface in patches] + [(2, face) for face, _ in wall_faces.values()]
        if fraction > 0:
            surfaces_in += [(2, tile) for tile in wall_tiles(occ, spec["dimensions_m"], fraction,
                                                             spec["absorption"]["rho_c_tile_m"], traps)]
        air, _ = occ.fragment(air, surfaces_in)
        occ.synchronize()
        volumes = [tag for dim, tag in air if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"the room's air is {len(volumes)} volumes, not one")
        volume = volumes[0]
        port_names = [port["name"] for port, _ in patches]
        trap_groups = [trap_group(trap) for trap in traps]
        groups = {name: [] for name in port_names + ["box", "walls", "patches"] + trap_groups}
        # the traps' faces (by centre AND area) and the corner prisms' rigid ends (by centre)
        trap_faces = ([(trap_group(trap), face_centre, trap["width_m"] * trap["height_m"])
                       for trap in traps if trap["mount"] == "corner"
                       for _, face_centre, _ in [prisms[trap["name"]]]]
                      + [(trap_group(trap), centre, trap["width_m"] * trap["height_m"])
                         for trap in traps if trap["mount"] == "wall" for _, centre in [wall_faces[trap["name"]]]])
        prism_ends = [end for _, _, end in prisms.values()]
        for _, surface in gmsh.model.getBoundary([(3, volume)], oriented=False):
            com, area = occ.getCenterOfMass(2, surface), occ.getMass(2, surface)
            trap_face = [group for group, centre, face_area in trap_faces
                         if math.dist(com, centre) < 1e-6 and abs(area - face_area) < 1e-6 * face_area]
            if trap_face:
                groups[trap_face[0]].append(surface)
                continue
            if any(math.dist(com, end) < 1e-6 for end in prism_ends):
                groups["walls"].append(surface)
                continue
            sx, sy, sz = com
            on_wall = min(abs(sx), abs(sx - lx), abs(sy), abs(sy - ly), abs(sz), abs(sz - lz)) < ON_SURFACE_M
            if on_wall:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surface)
                whole = sum(1 for lo, hi, length in ((xmin, xmax, lx), (ymin, ymax, ly), (zmin, zmax, lz))
                            if abs(hi - lo - length) < 1e-6) == 2
                groups["walls" if whole or fraction == 0 else "patches"].append(surface)
                continue
            groups[surface_of_boxes(boxes, com, area, in_box_frame)].append(surface)
        if fraction > 0 and not groups["patches"]:
            raise RuntimeError("patches were asked for and none was found on the walls")
        for name in port_names + trap_groups:
            if len(groups[name]) != 1:
                raise RuntimeError(f"(╯°□°)╯ {name} is {len(groups[name])} surfaces, not one")
        points = embedded_points(spec)
        point_tags = [occ.addPoint(*p) for p in points.values()]
        occ.synchronize()
        gmsh.model.mesh.embed(0, point_tags, 3, volume)
        for name, surfaces in groups.items():
            if surfaces:
                gmsh.model.addPhysicalGroup(2, surfaces, name=name)
        gmsh.model.addPhysicalGroup(3, [volume], name="air")
        refined_surfaces = [s for name in port_names + ["box"] + trap_groups for s in groups[name]]   # h_near_m
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", refined_surfaces)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", spec["h_near_m"])
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", spec["h_far_m"])
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.05)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.6)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        for option, value in (("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                              ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.ElementOrder", 2),
                              ("Mesh.MshFileVersion", 4.1), ("Mesh.SaveAll", 0)):
            gmsh.option.setNumber(option, value)
        gmsh.model.mesh.generate(3)
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        tetrahedra = sum(len(t) for t in element_tags)
        gmsh.write(str(path))
        area = {name: sum(occ.getMass(2, s) for s in surfaces) for name, surfaces in groups.items()}
        return {"nodes": len(node_tags), "tetrahedra": tetrahedra, "element_types": [int(t) for t in element_types],
                "area_m2": area, "yaw_deg": {box["name"]: math.degrees(yaw_of(box)) for box in boxes},
                "seat_points_m": seat_points(spec), "trap_points_m": trap_points(spec)}
    finally:
        gmsh.finalize()


def trap_prism(occ, trap, dimensions_m):
    """A corner trap's prism (spec.trap_face_frame) -> (volume tag, its face's centre, its room-side end's centre:
    the top of a floor prism, the bottom of a ceiling one)."""
    frame = trap_face_frame(trap, dimensions_m)
    (cx, cy), (sx, sy), leg = frame["corner_xy"], frame["into_room_xy"], frame["leg_m"]
    low, high = frame["z_range_m"]
    corners = [(cx, cy), (cx + sx * leg, cy), (cx, cy + sy * leg)]
    points = [occ.addPoint(x, y, low) for x, y in corners]
    lines = [occ.addLine(points[k], points[(k + 1) % 3]) for k in range(3)]
    triangle = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    prism = next(tag for dim, tag in occ.extrude([(2, triangle)], 0, 0, high - low) if dim == 3)
    end_z = high if trap["level"] == "floor" else low
    return prism, tuple(frame["face_centre_m"]), (cx + sx * leg / 3, cy + sy * leg / 3, end_z)


def wall_face(occ, trap, dimensions_m):
    """A wall trap's face: a rectangle in its room surface -> (surface tag, its centre)."""
    axis, plane, ((w0, w1), (h0, h1)), (width_axis, height_axis) = wall_face_rectangle(trap, dimensions_m)
    corners = []
    for w, h in ((w0, h0), (w1, h0), (w1, h1), (w0, h1)):
        point = [0.0, 0.0, 0.0]
        point[axis], point[width_axis], point[height_axis] = plane, w, h
        corners.append(tuple(point))
    return planar_quad(occ, corners), tuple(trap["centre_m"])


def wall_tiles(occ, dimensions_m, fraction, tile_m, traps):
    """The walls' rho c tiles (spec.rho_c_tiles: equal cells, one centred rectangle each, none touching a trap) as
    gmsh rectangles on the six surfaces."""
    tiles = []
    for wall, rectangles in rho_c_tiles(dimensions_m, fraction, tile_m, traps).items():
        axis, far = SURFACES[wall]
        at = dimensions_m[axis] if far else 0.0
        for (u0, u1), (v0, v1) in rectangles:
            rect = occ.addRectangle(u0, v0, 0, u1 - u0, v1 - v0)
            # the rectangle is made in the xy plane: move its (u, v, normal) to the wall's axes
            if axis == 0:     # x wall: u = y, v = z
                occ.rotate([(2, rect)], 0, 0, 0, 1, 1, 1, 2 * math.pi / 3)   # x->y, y->z, z->x
            elif axis == 1:   # y wall: u = x, v = z
                occ.rotate([(2, rect)], 0, 0, 0, 1, 0, 0, math.pi / 2)       # y->z
            occ.translate([(2, rect)], *[at if k == axis else 0.0 for k in range(3)])
            tiles.append(rect)
    return tiles


def kept_nodes(spec, mesh_path, facts):
    """The coordinates of the nodes a field-plane stage keeps (the module docstring), (node, 3)."""
    room_mesh = meshio.read(mesh_path)
    if set(room_mesh.cells_dict) - {"tetra10", "triangle6", "vertex"} or "tetra10" not in room_mesh.cells_dict:
        raise RuntimeError(f"(╯°□°)╯ {mesh_path}: cells {sorted(room_mesh.cells_dict)}, not a P2 room mesh")
    p2 = P2Mesh.from_arrays(room_mesh.points, room_mesh.cells_dict["tetra10"])
    keep = []
    for plane in spec["field_planes"]:
        grid = field_plane_points(spec, plane["name"])
        keep.append(p2.nodes_of(grid["points_m"][grid["in_air"]]))
    points = np.array(list(facts["seat_points_m"].values()) + list(facts["trap_points_m"].values()))
    keep.append(p2.nodes_of(points))
    faces = [name for name in facts["area_m2"] if facts["area_m2"][name] and name not in ("box", "walls", "patches")]
    for group in faces:
        triangles = room_mesh.cells_dict["triangle6"][room_mesh.cell_sets_dict[group]["triangle6"]]
        keep.append(np.unique(triangles))
    return room_mesh.points[np.unique(np.concatenate(keep))]


def boundary(group, kind, mesh_id, region_id):
    return {"group": {"dimension": 2, "mesh_id": mesh_id, "name": group, "tag": None}, "id": f"boundary:{group}",
            "kind": kind, "name": group, "parameters": {}, "region_id": region_id}


def velocity_port_parts(groups, mesh_id, region_id):
    """A PORT per physical group: its boundary ("moving"), component (ideal velocity, 1 m/s uniform) and excitation
    port "excitation:<group>" -- the sources' ports and the velocity_port trap faces alike."""
    boundaries = [boundary(group, "moving", mesh_id, region_id) for group in groups]
    components = [{"boundary_ids": [f"boundary:{group}"], "id": f"component:{group}", "kind": "ideal_velocity_source",
                   "name": f"{group} piston", "parameters": {"motion_profile": "uniform",
                                                            "boundary_motion_weights": {f"boundary:{group}": 1.0}}}
                  for group in groups]
    ports = [{"component_id": f"component:{group}", "id": f"excitation:{group}", "kind": "normal_velocity",
              "name": f"{group} unit normal velocity"} for group in groups]
    return boundaries, components, ports


def source_system_parts(boxes, mesh_id, region_id):
    """Every box's ports (velocity_port_parts) and the rigid rest of the boxes."""
    boundaries, components, ports = velocity_port_parts([port["name"] for box in boxes for port in box["ports"]],
                                                        mesh_id, region_id)
    return boundaries + [boundary("box", "rigid", mesh_id, region_id)], components, ports


def project_document(name, system, mesh_file, frequencies, components):
    return {"channel_config_by_name": {"main": {"delay_ms": 0.0, "hpf": {}, "level_db": 0.0, "lpf": {},
                                                "polarity": 1}},
            "component_channel_by_id": {component["id"]: "main" for component in components},
            "imported_meshes": [{"cleaned_file": None, "enabled": True, "name": name, "scale_factor": 1.0,
                                 "source_file": mesh_file, "translation_mm": [0.0, 0.0, 0.0]}],
            "physical_system": system,
            "project_preferences": {"freq_count": len(frequencies), "freq_max_hz": max(frequencies),
                                    "freq_min_hz": min(frequencies), "polar_angle_step_deg": 5.0,
                                    "polar_observation_distance_m": 1.0, "spherical_sampling_enabled": False},
            "schema_version": 9, "stitch_exterior_meshes": False, "symmetry": "off"}


def project(spec, mesh_facts):
    boundaries, components, ports = source_system_parts(checked_sources(spec), "mesh:room", "region:room")
    boundaries = [b for b in boundaries if mesh_facts["area_m2"].get(b["name"])]
    traps = checked_traps(spec)
    trap_boundaries, trap_components, trap_ports = velocity_port_parts(
        [trap_group(trap) for trap in traps if trap["face"] == "velocity_port"], "mesh:room", "region:room")
    boundaries += trap_boundaries + [boundary(trap_group(trap), "plane_wave_tube_termination", "mesh:room",
                                              "region:room") for trap in traps if trap["face"] == "rho_c"]
    components, ports = components + trap_components, ports + trap_ports
    for name in ("walls", "patches"):
        if mesh_facts["area_m2"].get(name):
            boundaries.append(boundary(name, "rigid" if name == "walls" else "plane_wave_tube_termination",
                                       "mesh:room", "region:room"))
    system = {
        "boundaries": boundaries, "components": components, "excitation_ports": ports,
        "id": f"system:room-{spec['name']}", "interfaces": [],
        "meshes": [{"file": MESH_FILE, "id": "mesh:room", "name": "room", "purpose": "fem_volume",
                    "scale_to_m": 1.0, "translation_m": [0.0, 0.0, 0.0]}],
        "metadata": {}, "model_version": 1, "name": f"room {spec['name']}",
        "regions": [{"density_kg_per_m3": DENSITY_KG_PER_M3, "id": "region:room", "kind": "bounded_air",
                     "loss_model": {"bulk_loss_factor": float(spec["absorption"]["bulk_loss_factor"])},
                     "mesh_ids": ["mesh:room"], "name": "room", "sound_speed_m_per_s": SOUND_SPEED_M_PER_S,
                     "volume_groups": [{"dimension": 3, "mesh_id": "mesh:room", "name": "air", "tag": None}]}]}
    return project_document("room", system, MESH_FILE, spec["frequencies_hz"], components)


def main(stage, spec_path):
    stage, spec = Path(stage), json.loads(Path(spec_path).read_text())
    build = stage / "build"
    build.mkdir(parents=True, exist_ok=False)
    facts = mesh(spec, build / MESH_FILE)
    if spec.get("field_planes"):
        keep = kept_nodes(spec, build / MESH_FILE, facts)
        np.save(build / KEPT_NODES_FILE, keep)
        shutil.copy(REDUCER_SCRIPT, build / REDUCER_SCRIPT.name)
        facts["kept_nodes"] = len(keep)
    (build / PROJECT_FILE).write_text(json.dumps(project(spec, facts), indent=1))
    (stage / REQUEST_FILE).write_text(json.dumps({"schema_version": 1, "frequencies_hz": spec["frequencies_hz"],
                                                  "retain": ["fem_nodal_pressure"]}, indent=1))
    (stage / SPEC_FILE).write_text(json.dumps(spec, indent=1))
    (stage / POINTS_FILE).write_text(json.dumps(facts, indent=1))
    print(json.dumps({k: v for k, v in facts.items() if k not in ("seat_points_m", "trap_points_m")}, indent=1))
    return stage


def in_first_frame(spec):
    """frame_of(box, points) for points given in the first box's frame: they are moved to the room (first box's
    pose), then into that box's frame."""
    first = checked_sources(spec)[0]
    yaw, (cx, cy, cz) = yaw_of(first), first["centre_m"]
    c, s = math.cos(yaw), math.sin(yaw)

    def to_room(points):
        return [[cx + x * c - y * s, cy + x * s + y * c, cz + z] for x, y, z in points]

    def to_first(points):
        return in_box_frame(first, points)

    def frame_of(box, points):
        return in_box_frame(box, to_room(points))
    return to_room, to_first, frame_of


def free_field_mesh(spec, path):
    """Every box's closed surface with its port patches, in the first box's frame, triangles, normals out of each body
    into the air (Boundary Lab's exterior convention; monitoring's model/acoustics/tests/test_bem_orientation.py proves
    it on a solve), checked triangle by triangle and by the signed volume."""
    boxes = checked_sources(spec)
    to_room, to_first, frame_of = in_first_frame(spec)
    first = boxes[0]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("boxes")
        occ = gmsh.model.occ
        bodies, centres = [], []
        for box in boxes:
            volume, patches = source_shapes(occ, box)
            moved = [(3, volume)] + [(2, surface) for _, surface in patches]
            occ.rotate(moved, 0, 0, 0, 0, 0, 1, yaw_of(box) - yaw_of(first))
            centre = to_first([box["centre_m"]])[0]
            occ.translate(moved, *centre)
            body, _ = occ.fragment([(3, volume)], [(2, surface) for _, surface in patches])
            bodies += [tag for dim, tag in body if dim == 3]
            centres.append(centre)
        occ.synchronize()
        names = [port["name"] for box in boxes for port in box["ports"]]
        groups = {name: [] for name in names + ["box"]}
        owner = {}
        for body, centre in zip(bodies, centres):
            for _, surface in gmsh.model.getBoundary([(3, body)], oriented=False):
                owner[surface] = centre
                groups[surface_of_boxes(boxes, occ.getCenterOfMass(2, surface), occ.getMass(2, surface),
                                        frame_of)].append(surface)
        for name in names:
            if len(groups[name]) != 1:
                raise RuntimeError(f"(╯°□°)╯ port {name} is {len(groups[name])} surfaces of the boxes, not one")
        for name, surfaces in groups.items():
            gmsh.model.addPhysicalGroup(2, surfaces, name=name)
        gmsh.option.setNumber("Mesh.MeshSizeMin", FREE_FIELD_ELEMENT_M)
        gmsh.option.setNumber("Mesh.MeshSizeMax", FREE_FIELD_ELEMENT_M)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.model.mesh.generate(2)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        where = {int(t): coordinates[3 * i:3 * i + 3] for i, t in enumerate(node_tags)}
        surfaces = [tag for name in groups for tag in groups[name]]
        for surface in surfaces:                         # a convex box: out of the body = away from its own centre
            _, _, nodes = gmsh.model.mesh.getElements(2, surface)
            first_triangle = [where[int(t)] for t in nodes[0][:3]]
            normal = cross(sub(first_triangle[1], first_triangle[0]), sub(first_triangle[2], first_triangle[0]))
            centroid = sub([sum(c[k] for c in first_triangle) / 3 for k in range(3)], owner[surface])
            if sum(n * c for n, c in zip(normal, centroid)) < 0:
                gmsh.model.mesh.reverse([(2, surface)])
        signed_volume, inward = 0.0, 0
        for surface in surfaces:
            _, _, nodes = gmsh.model.mesh.getElements(2, surface)
            tags = nodes[0]
            for i in range(0, len(tags), 3):
                p, q, r = (where[int(t)] for t in tags[i:i + 3])
                normal = cross(sub(q, p), sub(r, p))
                centroid = sub([(p[k] + q[k] + r[k]) / 3 for k in range(3)], owner[surface])
                inward += sum(n * c for n, c in zip(normal, centroid)) <= 0
                signed_volume += sum(pk * ck for pk, ck in zip(p, cross(q, r))) / 6
        expected = sum(math.prod(box["size_m"]) for box in boxes)
        if inward or abs(signed_volume - expected) > 1e-6 * expected:
            raise RuntimeError(f"(╯°□°)╯ the boxes' surfaces have {inward} inward triangles, signed volume "
                               f"{signed_volume} against {expected}")
        gmsh.write(str(path))
        triangles = sum(len(t) for t in gmsh.model.mesh.getElements(2)[1])
        return {"nodes": len(node_tags), "triangles": triangles, "signed_volume_m3": signed_volume,
                "area_m2": {name: sum(occ.getMass(2, s) for s in tags) for name, tags in groups.items()}}
    finally:
        gmsh.finalize()


def sub(p, q):
    return [p[k] - q[k] for k in range(3)]


def cross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def free_field_project(name, boxes, frequencies):
    boundaries, components, ports = source_system_parts(boxes, "mesh:box", "region:air")
    system = {
        "boundaries": boundaries, "components": components, "excitation_ports": ports,
        "id": f"system:freefield-{name}", "interfaces": [],
        "meshes": [{"file": FREE_FIELD_MESH_FILE, "id": "mesh:box", "name": "box", "purpose": "bem_surface",
                    "scale_to_m": 1.0, "translation_m": [0.0, 0.0, 0.0]}],
        "metadata": {}, "model_version": 1, "name": f"free field {name}",
        "regions": [{"density_kg_per_m3": DENSITY_KG_PER_M3, "id": "region:air", "kind": "unbounded_air",
                     "loss_model": {}, "mesh_ids": ["mesh:box"], "name": "air",
                     "sound_speed_m_per_s": SOUND_SPEED_M_PER_S, "volume_groups": []}]}
    return project_document("box", system, FREE_FIELD_MESH_FILE, frequencies, components)


def free_field(stage, spec_paths):
    stage = Path(stage)
    specs = [json.loads(Path(p).read_text()) for p in spec_paths]
    first, frequencies = specs[0], specs[0]["frequencies_hz"]
    for spec in specs:
        if the_boxes(spec) != the_boxes(first) or spec["frequencies_hz"] != frequencies:
            raise RuntimeError(f"(╯°□°)╯ {spec['name']}: other boxes (or the same standing otherwise) or another "
                               f"frequency grid than {first['name']}'s")
    build = stage / "build"
    build.mkdir(parents=True, exist_ok=False)
    facts = free_field_mesh(first, build / FREE_FIELD_MESH_FILE)
    boxes = checked_sources(first)
    (build / FREE_FIELD_PROJECT_FILE).write_text(json.dumps(free_field_project(stage.name, boxes, frequencies),
                                                            indent=1))
    probes = [{"id": spec["name"], "coordinate_frame": "project",
               "points_m": in_box_frame(checked_sources(spec)[0], list(seat_points(spec).values()))}
              for spec in specs]
    to_room, to_first, _ = in_first_frame(first)
    extra = {}
    for box in boxes:
        yaw, (cx, cy, cz) = yaw_of(box), box["centre_m"]
        c, s = math.cos(yaw), math.sin(yaw)
        for point_name, (x, y, z) in (box.get("free_field_points_m") or {}).items():
            extra[f"{box['name']}/{point_name}"] = to_first([[cx + x * c - y * s, cy + x * s + y * c, cz + z]])[0]
    if extra:
        probes.append({"id": FREE_FIELD_BOX_PROBE, "coordinate_frame": "project", "points_m": list(extra.values())})
    (stage / REQUEST_FILE).write_text(json.dumps({"schema_version": 1, "frequencies_hz": frequencies,
                                                  "probes": probes, "include_project_observations": False},
                                                 indent=1))
    (stage / FREE_FIELD_SPECS_FILE).write_text(json.dumps(
        {"sources": the_boxes(first), "probes": {**{spec["name"]: list(seat_points(spec)) for spec in specs},
                                                 **({FREE_FIELD_BOX_PROBE: list(extra)} if extra else {})}},
        indent=1))
    print(json.dumps(facts, indent=1))
    return stage


def submit(stage, expected_s, ram_gb, cpu):
    """The JOB (acoustic_solve.solve_job): CPU only (requires.gpu "none": interior FEM and this BEM run on the CPU
    whatever the backend, monitoring's PLANS/2026-10-14-room-cardioid-sub.md), `cpu` cores, which are also its Julia
    and BLAS threads; resumable per frequency (2026-10-16: four chunks killed at 103-109 of 115 frequencies were
    solved again from scratch)."""
    stage = Path(stage)
    projects = sorted(path.name for path in (stage / "build").glob("*.blab.json"))
    if len(projects) != 1:
        raise RuntimeError(f"(╯°□°)╯ {stage}/build holds {projects}, not one project")
    return solve_job.queue_solve(stage, f"build/{projects[0]}", REQUEST_FILE, "beat_cpu",
                                 resources={"cpu": cpu, "gpu_memory_gb": 0, "ram_gb": ram_gb}, expected_s=expected_s,
                                 label=f"{stage.name} room {projects[0]}", threads=math.ceil(cpu), resumable=True,
                                 requires={"gpu": "none"})


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "submit":
        submit(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
    elif len(sys.argv) >= 4 and sys.argv[1] == "freefield":
        free_field(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
