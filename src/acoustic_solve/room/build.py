"""A ROOM: a closed shoebox of air with a source box in it, as a Boundary Lab INTERIOR FEM project, staged as a podd
JOB. Its first studies were the speaker project's (not public: PLANS/2026-10-14-room-cardioid-sub.md,
two pistons on a cube; PLANS/2026-10-15-room-a2-crossover.md, A2's box and its four sources).

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
    mesh nodes and read out of the retained fem_nodal_pressure.

THE SOURCE BOX (`source`): a rigid rectangular box, `size_m` [depth, width, height], its centre at `centre_m`, yawed
so its front face (+depth) points at `aim_m` (a horizontal point). Its BOX FRAME: x along the depth (front face at
+depth/2), y across (left face at +width/2), z up. Each of its `ports` is a patch on one face, its own ideal-velocity
component and excitation port `excitation:<name>` (1 m/s uniform normal velocity, along the FEM domain's outward
normal, i.e. INTO the box), so any mix of sources is a post-processing weight (acoustic_solve.weighted_sums,
acoustic_solve.room.analysis). A port: {"name", "face": front | rear | left | right, "shape": "face" (the whole face) |
"disc" ("radius_m") | "rect" ("size_m": [h, v]), "at_m": [h, v]} where (h, v) are the face's in-plane coordinates of
the patch centre: (y, z) on front / rear, (x, z) on left / right. Everything of the box that is no port is rigid.

SEATS (`seats_m`: {seat: [x, y, z]}): each a grid at ear height (a square of (2n+1)^2 points, `seat_grid`) and the
centre 0.2 m lower and higher, named "<seat>/x+0y+0", "<seat>/z-0.2", ...; every point an embedded mesh node.

TRAPS (optional, `traps`): an active bass trap in a room corner (the active bass trap project), modelled as an
IDEAL BROADBAND ABSORBER: a triangular prism standing on the floor in the corner (cut out of the air), its 45-degree
face `width_m` wide and `height_m` tall a plane-wave (rho c) termination like the wall patches, its top rigid. A real
active trap works over a band (AVAA ~30-150 Hz); this one absorbs at every frequency: a known simplification.

THE FREE FIELD: the room's effect on a source mix is its seat pressure over the SAME box's anechoic pressure at the
same points, so one exterior BEM of the box alone (every port, rigid elsewhere) is solved with the seat points of
every room spec given, moved into the box frame, as point probes, one probe per spec named after it; plus the
source's `free_field_points_m` ({name: [x, y, z]} in the box frame, e.g. an on-axis point), probe "box". A rigid
yawed translation is all a placement is, so one solve serves every spec of the same box.

<room.json>: {"name", "dimensions_m": [Lx, Ly, Lz], "source": {"size_m", "centre_m", "aim_m", "ports",
optional "free_field_points_m"}, "seats_m", "seat_grid": {"step_m", "half_count"}, "h_far_m", "h_near_m",
"absorption": {"bulk_loss_factor": eta, "rho_c_patch_fraction": f (0: rigid walls)}, "frequencies_hz": [...],
optional "traps": [{"corner": [i, j] (0: the x = 0 / y = 0 wall, 1: the far one), "width_m", "height_m"}]}
"""
import json
import math
import sys
from pathlib import Path

import gmsh

from acoustic_solve import solve_job
from acoustic_solve.room.spec import (DENSITY_KG_PER_M3, FREE_FIELD_BOX_PROBE, FREE_FIELD_SPECS_FILE, POINTS_FILE,
                                      SOUND_SPEED_M_PER_S, in_box_frame, seat_points, the_box, yaw_of)

MESH_FILE = "room.msh"
PROJECT_FILE = "room.blab.json"
FREE_FIELD_MESH_FILE = "box.msh"
FREE_FIELD_PROJECT_FILE = "freefield.blab.json"
FREE_FIELD_ELEMENT_M = 0.025         # lambda / 27 at 500 Hz
REQUEST_FILE = "request.json"
FACES = ("front", "rear", "left", "right")
ON_SURFACE_M = 1e-6


def port_geometry(source, port):
    """(face normal axis, sign, the patch's centre in the box frame, its in-plane half-extents, its area)."""
    depth, width, height = source["size_m"]
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


def source_shapes(occ, source):
    """The box and its port patches at the box frame's origin -> (box volume tag, [(port, surface tag)])."""
    depth, width, height = source["size_m"]
    box = occ.addBox(-depth / 2, -width / 2, -height / 2, depth, width, height)
    patches = []
    for port in source["ports"]:
        if port["face"] not in FACES:
            raise RuntimeError(f"(╯°□°)╯ port {port['name']}: face {port['face']!r} is not one of {FACES}")
        axis, sign, (cx, cy, cz), half_h, half_v, _ = port_geometry(source, port)
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
            tags = [occ.addPoint(*corner) for corner in corners]
            lines = [occ.addLine(tags[k], tags[(k + 1) % 4]) for k in range(4)]
            surface = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        patches.append((port, surface))
    return box, patches


def classify_source_surface(source, point_box_frame, area):
    """A surface of the box (its centre of mass in the box frame, its area) -> the port it is, or "box". A port is
    told by its centre AND its area: a face with a port cut out of it has the face's centre, not the port's area."""
    found = []
    for port in source["ports"]:
        axis, _, centre, half_h, half_v, port_area = port_geometry(source, port)
        h_axis = 1 if axis == 0 else 0
        if (abs(point_box_frame[axis] - centre[axis]) < 1e-6
                and abs(point_box_frame[h_axis] - centre[h_axis]) <= half_h + 1e-6
                and abs(point_box_frame[2] - centre[2]) <= half_v + 1e-6
                and abs(area - port_area) < 2e-3 * port_area):
            found.append(port["name"])
    if len(found) > 1:
        raise RuntimeError(f"(╯°□°)╯ a source surface at {point_box_frame} matches ports {found}")
    return found[0] if found else "box"


def on_box(source, point_box_frame):
    depth, width, height = source["size_m"]
    x, y, z = point_box_frame
    inside = abs(x) <= depth / 2 + 1e-6 and abs(y) <= width / 2 + 1e-6 and abs(z) <= height / 2 + 1e-6
    return inside and min(abs(abs(x) - depth / 2), abs(abs(y) - width / 2), abs(abs(z) - height / 2)) < 1e-6


def mesh(spec, path):
    lx, ly, lz = spec["dimensions_m"]
    source = spec["source"]
    yaw = yaw_of(source)
    occ = gmsh.model.occ
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("room")
        room = occ.addBox(0, 0, 0, lx, ly, lz)
        box, patches = source_shapes(occ, source)
        moved = [(3, box)] + [(2, surface) for _, surface in patches]
        occ.rotate(moved, 0, 0, 0, 0, 0, 1, yaw)
        occ.translate(moved, *source["centre_m"])
        traps = [trap_prism(occ, trap, lx, ly) for trap in spec.get("traps", [])]
        air, _ = occ.cut([(3, room)], [(3, box)] + [(3, prism) for prism, _, _ in traps])
        fraction = spec["absorption"]["rho_c_patch_fraction"]
        surfaces_in = [(2, surface) for _, surface in patches]
        if fraction > 0:
            surfaces_in += [(2, p) for p in wall_patches(occ, lx, ly, lz, fraction)]
        air, _ = occ.fragment(air, surfaces_in)
        occ.synchronize()
        volumes = [tag for dim, tag in air if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"the room's air is {len(volumes)} volumes, not one")
        volume = volumes[0]
        port_names = [port["name"] for port in source["ports"]]
        groups = {name: [] for name in port_names + ["box", "walls", "patches", "traps"]}
        for _, surface in gmsh.model.getBoundary([(3, volume)], oriented=False):
            sx, sy, sz = occ.getCenterOfMass(2, surface)
            trap_face = next((kind for _, face_centre, top_centre in traps for kind, centre in
                              (("traps", face_centre), ("walls", top_centre))
                              if math.dist((sx, sy, sz), centre) < 1e-6), None)
            if trap_face:                                # the absorbing face, or the rigid top
                groups[trap_face].append(surface)
                continue
            on_wall = min(abs(sx), abs(sx - lx), abs(sy), abs(sy - ly), abs(sz), abs(sz - lz)) < ON_SURFACE_M
            if on_wall:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surface)
                whole = sum(1 for lo, hi, length in ((xmin, xmax, lx), (ymin, ymax, ly), (zmin, zmax, lz))
                            if abs(hi - lo - length) < 1e-6) == 2
                groups["walls" if whole or fraction == 0 else "patches"].append(surface)
                continue
            at = in_box_frame(source, [[sx, sy, sz]])[0]
            if not on_box(source, at):
                raise RuntimeError(f"(╯°□°)╯ a boundary surface at {(sx, sy, sz)} is on no wall, trap or the box")
            groups[classify_source_surface(source, at, occ.getMass(2, surface))].append(surface)
        if fraction > 0 and not groups["patches"]:
            raise RuntimeError("patches were asked for and none was found on the walls")
        if len(groups["traps"]) != len(traps):
            raise RuntimeError(f"(╯°□°)╯ {len(traps)} traps asked for, {len(groups['traps'])} absorbing faces found")
        for name in port_names:
            if len(groups[name]) != 1:
                raise RuntimeError(f"(╯°□°)╯ port {name} is {len(groups[name])} surfaces, not one")
        points = seat_points(spec)
        point_tags = [occ.addPoint(*p) for p in points.values()]
        occ.synchronize()
        gmsh.model.mesh.embed(0, point_tags, 3, volume)
        for name, surfaces in groups.items():
            if surfaces:
                gmsh.model.addPhysicalGroup(2, surfaces, name=name)
        gmsh.model.addPhysicalGroup(3, [volume], name="air")
        source_surfaces = [s for name in port_names + ["box"] for s in groups[name]]
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", source_surfaces)
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
                "area_m2": area, "yaw_deg": math.degrees(yaw), "seat_points_m": points}
    finally:
        gmsh.finalize()


def trap_prism(occ, trap, lx, ly):
    """A floor-standing triangular prism in a corner: legs along the two walls, its 45-degree face `width_m` wide,
    `height_m` tall -> (volume tag, the absorbing face's centre, the top's centre)."""
    i, j = trap["corner"]
    leg = trap["width_m"] / math.sqrt(2)
    cx, cy = (0.0, lx)[i], (0.0, ly)[j]
    sx, sy = (1.0, -1.0)[i], (1.0, -1.0)[j]                       # into the room
    corners = [(cx, cy), (cx + sx * leg, cy), (cx, cy + sy * leg)]
    points = [occ.addPoint(x, y, 0.0) for x, y in corners]
    lines = [occ.addLine(points[k], points[(k + 1) % 3]) for k in range(3)]
    triangle = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    height = trap["height_m"]
    prism = next(tag for dim, tag in occ.extrude([(2, triangle)], 0, 0, height) if dim == 3)
    face_centre = (cx + sx * leg / 2, cy + sy * leg / 2, height / 2)
    top_centre = (cx + sx * leg / 3, cy + sy * leg / 3, height)
    return prism, face_centre, top_centre


def wall_patches(occ, lx, ly, lz, fraction):
    """A centred rectangle covering `fraction` of each of the six surfaces (sides scaled by sqrt(fraction))."""
    s = math.sqrt(fraction)
    patches = []
    for axis, length, (u, v) in ((0, lx, (ly, lz)), (1, ly, (lx, lz)), (2, lz, (lx, ly))):
        for at in (0.0, length):
            du, dv = u * s, v * s
            rect = occ.addRectangle((u - du) / 2, (v - dv) / 2, 0, du, dv)
            # the rectangle is made in the xy plane: move its (u, v, normal) to the wall's axes
            if axis == 0:     # x wall: u = y, v = z
                occ.rotate([(2, rect)], 0, 0, 0, 1, 1, 1, 2 * math.pi / 3)   # x->y, y->z, z->x
            elif axis == 1:   # y wall: u = x, v = z
                occ.rotate([(2, rect)], 0, 0, 0, 1, 0, 0, math.pi / 2)       # y->z
            occ.translate([(2, rect)], *[at if k == axis else 0.0 for k in range(3)])
            patches.append(rect)
    return patches


def source_system_parts(source, mesh_id, region_id):
    """The ports' boundaries ("moving"), components (ideal velocity, 1 m/s) and excitation ports, and the rigid rest."""
    names = [port["name"] for port in source["ports"]]
    boundaries = [{"group": {"dimension": 2, "mesh_id": mesh_id, "name": name, "tag": None},
                   "id": f"boundary:{name}", "kind": "moving" if name in names else "rigid", "name": name,
                   "parameters": {}, "region_id": region_id} for name in names + ["box"]]
    components = [{"boundary_ids": [f"boundary:{name}"], "id": f"component:{name}", "kind": "ideal_velocity_source",
                   "name": f"{name} piston", "parameters": {"motion_profile": "uniform",
                                                           "boundary_motion_weights": {f"boundary:{name}": 1.0}}}
                  for name in names]
    ports = [{"component_id": f"component:{name}", "id": f"excitation:{name}", "kind": "normal_velocity",
              "name": f"{name} unit normal velocity"} for name in names]
    return boundaries, components, ports


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
    boundaries, components, ports = source_system_parts(spec["source"], "mesh:room", "region:room")
    boundaries = [b for b in boundaries if mesh_facts["area_m2"].get(b["name"])]
    for name in ("walls", "patches", "traps"):
        if mesh_facts["area_m2"].get(name):
            boundaries.append({"group": {"dimension": 2, "mesh_id": "mesh:room", "name": name, "tag": None},
                               "id": f"boundary:{name}", "name": name, "parameters": {}, "region_id": "region:room",
                               "kind": "rigid" if name == "walls" else "plane_wave_tube_termination"})
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
    (build / PROJECT_FILE).write_text(json.dumps(project(spec, facts), indent=1))
    (stage / REQUEST_FILE).write_text(json.dumps({"schema_version": 1, "frequencies_hz": spec["frequencies_hz"],
                                                  "retain": ["fem_nodal_pressure"]}, indent=1))
    (stage / "room.json").write_text(json.dumps(spec, indent=1))
    (stage / POINTS_FILE).write_text(json.dumps(facts, indent=1))
    print(json.dumps({k: v for k, v in facts.items() if k != "seat_points_m"}, indent=1))
    return stage


def free_field_mesh(source, path):
    """The box's closed surface with its port patches, triangles, normals out of the body into the air (Boundary
    Lab's exterior convention; monitoring's model/acoustics/tests/test_bem_orientation.py proves it on a solve), checked
    triangle by triangle and by the signed volume."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        occ = gmsh.model.occ
        box, patches = source_shapes(occ, source)
        body, _ = occ.fragment([(3, box)], [(2, surface) for _, surface in patches])
        occ.synchronize()
        volume = next(tag for dim, tag in body if dim == 3)
        names = [port["name"] for port in source["ports"]]
        groups = {name: [] for name in names + ["box"]}
        for _, surface in gmsh.model.getBoundary([(3, volume)], oriented=False):
            groups[classify_source_surface(source, occ.getCenterOfMass(2, surface), occ.getMass(2, surface))].append(
                surface)
        for name in names:
            if len(groups[name]) != 1:
                raise RuntimeError(f"(╯°□°)╯ port {name} is {len(groups[name])} surfaces of the box, not one")
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
        for surface in surfaces:                         # a convex box: out of the body = away from its centre
            _, _, nodes = gmsh.model.mesh.getElements(2, surface)
            first = [where[int(t)] for t in nodes[0][:3]]
            normal = cross(sub(first[1], first[0]), sub(first[2], first[0]))
            centroid = [sum(c[k] for c in first) / 3 for k in range(3)]
            if sum(n * c for n, c in zip(normal, centroid)) < 0:
                gmsh.model.mesh.reverse([(2, surface)])
        signed_volume, inward = 0.0, 0
        for surface in surfaces:
            _, _, nodes = gmsh.model.mesh.getElements(2, surface)
            tags = nodes[0]
            for i in range(0, len(tags), 3):
                p, q, r = (where[int(t)] for t in tags[i:i + 3])
                normal = cross(sub(q, p), sub(r, p))
                centroid = [(p[k] + q[k] + r[k]) / 3 for k in range(3)]
                inward += sum(n * c for n, c in zip(normal, centroid)) <= 0
                signed_volume += sum(pk * ck for pk, ck in zip(p, cross(q, r))) / 6
        expected = math.prod(source["size_m"])
        if inward or abs(signed_volume - expected) > 1e-6 * expected:
            raise RuntimeError(f"(╯°□°)╯ the box surface has {inward} inward triangles, signed volume "
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


def free_field_project(name, source, frequencies):
    boundaries, components, ports = source_system_parts(source, "mesh:box", "region:air")
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
    source, frequencies = specs[0]["source"], specs[0]["frequencies_hz"]
    for spec in specs:
        if the_box(spec["source"]) != the_box(source) or spec["frequencies_hz"] != frequencies:
            raise RuntimeError(f"(╯°□°)╯ {spec['name']}: another box or frequency grid than {specs[0]['name']}'s")
    build = stage / "build"
    build.mkdir(parents=True, exist_ok=False)
    facts = free_field_mesh(source, build / FREE_FIELD_MESH_FILE)
    (build / FREE_FIELD_PROJECT_FILE).write_text(json.dumps(free_field_project(stage.name, source, frequencies),
                                                            indent=1))
    probes = [{"id": spec["name"], "coordinate_frame": "project",
               "points_m": in_box_frame(spec["source"], list(seat_points(spec).values()))} for spec in specs]
    extra = source.get("free_field_points_m") or {}
    if extra:
        probes.append({"id": FREE_FIELD_BOX_PROBE, "coordinate_frame": "project", "points_m": list(extra.values())})
    (stage / REQUEST_FILE).write_text(json.dumps({"schema_version": 1, "frequencies_hz": frequencies,
                                                  "probes": probes, "include_project_observations": False},
                                                 indent=1))
    (stage / FREE_FIELD_SPECS_FILE).write_text(json.dumps(
        {"source": the_box(source), "probes": {**{spec["name"]: list(seat_points(spec)) for spec in specs},
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
