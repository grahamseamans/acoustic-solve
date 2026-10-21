"""A ROOM SPEC (room.json; acoustic_solve.room.build's docstring is its grammar) read without gmsh: the air, the source
boxes and their frames, the seat points and their names, the traps (their mounts, kinds, faces, frames, mic points and
names), the field planes. The builder writes these and the readers read them back, so both import them from here and
nothing else spells a seat point's, a trap point's, a trap group's or a sensed quantity's name.
"""
import math

import numpy as np

# THE AIR every room is solved in: dry air at 20 degC and 101325 Pa, from E. Sengpiel, "Comparative representation of
# sound field quantities" (sengpielaudio.com, mirrored by references/fetch_sources.sh), lines 6-8 of its text: "Air
# temperarature = 20 degC: (Atmospheric pressure 10,1325 Pa ...)", "Density of air at 20 degC: = 1.204 kg / m3",
# "Speed of sound at 20 degC: c = 343 m / s" (its rho c, line 10: 413 N s / m^3). (Until 2026-10-20 these were
# Boundary Lab's region defaults, 1.21 and 343.0, physical_model.py:96-97, which give no source.)
SOUND_SPEED_M_PER_S = 343.0
DENSITY_KG_PER_M3 = 1.204
POINTS_FILE = "seat_points.json"               # the room stage's: {"seat_points_m": {name: [x, y, z]},
                                               #   "trap_points_m": {"<trap>/<mic>": [x, y, z]}, mesh facts}
SPEC_FILE = "room.json"                        # the room stage's copy of its spec
MESH_FILE = "room.msh"                         # the room stage's mesh, in build/ (gmsh 4.1, P2)
KEPT_NODES_FILE = "keep_nodes.npy"             # build/'s: the coordinates of the nodes the pod keeps (field_planes)
FREE_FIELD_SPECS_FILE = "freefield.json"       # the free-field stage's: {"sources", "probes": {probe: [point names]}}
FREE_FIELD_BOX_PROBE = "box"                   # the probe of the boxes' own free_field_points_m
SEAT_CENTRE_POINT = "x+0y+0z+0"                # the seat itself, the centre of its box of points
SEAT_EAR_HEIGHT_SUFFIX = "z+0"                 # a seat's ear-height grid: its points at the centre's height
TRAP_FACE_KINDS = ("velocity_port", "rho_c")   # a trap face: a port (any law, post-processed) | a rho c termination
TRAP_MOUNTS = ("corner", "wall")               # a prism in a vertical corner | a face flush in a room surface
CORNER_LEVELS = ("floor", "ceiling")           # a corner prism stands on the floor or hangs from the ceiling
SURFACES = {"x0": (0, 0), "x1": (0, 1), "y0": (1, 0), "y1": (1, 1), "z0": (2, 0), "z1": (2, 1)}   # (axis, far side)
TRAP_FACE_SENSED = "face"                      # "<trap>/face": the face-averaged pressure, a sensed quantity
TRAP_GROUP_PREFIX = "trap:"                    # a trap face's physical group "trap:<name>"; its port is
                                               #   "excitation:trap:<name>" (a port id is "excitation:<group>")
IN_AIR_MARGIN_M = 1e-6                         # an embedded point this close to a boundary is not in the air
ON_SURFACE_M = 1e-6


def checked_sources(spec):
    """The spec's source boxes, refused loudly unless a non-empty list of {"name", "size_m", "centre_m", "aim_m",
    "ports", optional "free_field_points_m"} with unique box names and port names unique across every box."""
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"(╯°□°)╯ spec {spec.get('name')!r}: 'sources' must be a non-empty list of boxes "
                           f"(there is no single 'source' any more)")
    boxes, ports = [], []
    for box in sources:
        unknown = set(box) - {"name", "size_m", "centre_m", "aim_m", "ports", "free_field_points_m"}
        if unknown or not isinstance(box.get("name"), str) or not box["name"]:
            raise RuntimeError(f"(╯°□°)╯ source box {box.get('name')!r}: a name is required; unknown keys "
                               f"{sorted(unknown)}")
        boxes.append(box["name"])
        ports += [port["name"] for port in box["ports"]]
    if len(set(boxes)) != len(boxes) or len(set(ports)) != len(ports):
        raise RuntimeError(f"(╯°□°)╯ box names {boxes} or port names {ports} are not unique")
    return sources


def seat_points(spec):
    """Every seat's box of points: {"<seat>/x<i>y<j>z<k>": [x, y, z]} (signed steps from its centre_m along the room's
    axes, step_m [sx, sy, sz], i in -nx..nx for half_count [nx, ny, nz]); the centre is "x+0y+0z+0"."""
    points = {}
    for seat, box in spec["seats"].items():
        if set(box) != {"centre_m", "step_m", "half_count"}:
            raise RuntimeError(f"(╯°□°)╯ seat {seat}: keys {sorted(box)}, not centre_m, step_m, half_count")
        (x0, y0, z0), (sx, sy, sz), (nx, ny, nz) = box["centre_m"], box["step_m"], box["half_count"]
        points.update({f"{seat}/x{i:+d}y{j:+d}z{k:+d}": [x0 + i * sx, y0 + j * sy, z0 + k * sz]
                       for i in range(-nx, nx + 1) for j in range(-ny, ny + 1) for k in range(-nz, nz + 1)})
    return points


def seat_point_indices(point_names, seat):
    """(the seat's centre index, its ear-height grid's indices: its points at the centre's height) in a list of seat
    point names."""
    centre = f"{seat}/{SEAT_CENTRE_POINT}"
    if centre not in point_names:
        raise RuntimeError(f"(╯°□°)╯ no seat point {centre!r} among {len(point_names)} points")
    return point_names.index(centre), [i for i, name in enumerate(point_names)
                                       if name.startswith(f"{seat}/x") and name.endswith(SEAT_EAR_HEIGHT_SUFFIX)]


def seats_of(point_names):
    """The seats named in a list of seat point names, sorted."""
    return sorted({name.split("/")[0] for name in point_names})


def yaw_of(box):
    """The box's yaw: its front face toward aim_m."""
    return math.atan2(box["aim_m"][1] - box["centre_m"][1], box["aim_m"][0] - box["centre_m"][0])


def in_box_frame(box, points):
    """Room points -> the box frame (centre at the origin, front face along +x)."""
    yaw = yaw_of(box)
    cx, cy, cz = box["centre_m"]
    c, s = math.cos(yaw), math.sin(yaw)
    return [[(x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c, z - cz] for x, y, z in points]


def the_boxes(spec):
    """What makes two specs' free fields one: every box, its ports and free-field points, and where each box stands
    relative to the FIRST box (its centre in the first box's frame and its yaw against the first's, rounded to 1e-9),
    never where the first stands in the room."""
    sources = checked_sources(spec)
    first = sources[0]
    boxes = []
    for box in sources:
        centre = in_box_frame(first, [box["centre_m"]])[0]
        boxes.append({"name": box["name"], "size_m": box["size_m"], "ports": box["ports"],
                      "free_field_points_m": box.get("free_field_points_m"),
                      "centre_in_first_m": [round(c, 9) for c in centre],
                      "yaw_from_first_deg": round(math.degrees(yaw_of(box) - yaw_of(first)) % 360.0, 9)})
    return boxes


def checked_traps(spec):
    """The spec's traps (none: []), each refused loudly unless it is {"name", "face": one of TRAP_FACE_KINDS, "mount":
    one of TRAP_MOUNTS, "width_m", "height_m", optional "mic_points_m"} plus, for a corner mount, "corner" [i, j] and
    "level" (one of CORNER_LEVELS), for a wall mount "wall" (one of SURFACES) and "centre_m" (on that surface, the
    face inside it); unique names, mic names that cannot be mistaken for the face ("face") or split a sensed name
    ("/"), and no two wall faces on one surface overlapping."""
    traps = spec.get("traps", [])
    names = []
    for number, trap in enumerate(traps):
        name = trap.get("name")
        if not isinstance(name, str) or not name or "/" in name:
            raise RuntimeError(f"(╯°□°)╯ trap {number}: name {name!r} (required: a non-empty string without '/')")
        if trap.get("face") not in TRAP_FACE_KINDS:
            raise RuntimeError(f"(╯°□°)╯ trap {name}: face {trap.get('face')!r} is not one of {TRAP_FACE_KINDS} "
                               f"(required, no default)")
        mount = trap.get("mount")
        if mount not in TRAP_MOUNTS:
            raise RuntimeError(f"(╯°□°)╯ trap {name}: mount {mount!r} is not one of {TRAP_MOUNTS} (required, no "
                               f"default)")
        own = {"corner": {"corner", "level"}, "wall": {"wall", "centre_m"}}[mount]
        allowed = {"name", "face", "mount", "width_m", "height_m", "mic_points_m"} | own
        if set(trap) - allowed or not own <= set(trap):
            raise RuntimeError(f"(╯°□°)╯ trap {name} ({mount}): keys {sorted(trap)}; needs {sorted(own)}, allows "
                               f"{sorted(allowed)}")
        if mount == "corner" and trap["level"] not in CORNER_LEVELS:
            raise RuntimeError(f"(╯°□°)╯ trap {name}: level {trap['level']!r} is not one of {CORNER_LEVELS}")
        if mount == "wall":
            wall_face_rectangle(trap, spec["dimensions_m"])           # refuses a face off or past its surface
        for mic in trap.get("mic_points_m", {}):
            if not mic or "/" in mic or mic == TRAP_FACE_SENSED:
                raise RuntimeError(f"(╯°□°)╯ trap {name}: mic name {mic!r} (not empty, no '/', not "
                                   f"{TRAP_FACE_SENSED!r})")
        names.append(name)
    if len(set(names)) != len(names):
        raise RuntimeError(f"(╯°□°)╯ trap names {names} are not unique")
    walls = [trap for trap in traps if trap["mount"] == "wall"]
    for a in range(len(walls)):
        for b in range(a + 1, len(walls)):
            if rectangles_overlap(walls[a], walls[b], spec["dimensions_m"]):
                raise RuntimeError(f"(╯°□°)╯ wall traps {walls[a]['name']} and {walls[b]['name']} overlap")
    return traps


def trap_group(trap):
    """The physical group of a trap's face (its port, for a velocity_port face, is "excitation:" + this)."""
    return f"{TRAP_GROUP_PREFIX}{trap['name']}"


def surface_axes(wall):
    """A room surface's (normal axis, the two in-plane axes (width, height) of a face on it): a wall's width is
    horizontal and its height z; the floor's and ceiling's width is x and height y."""
    axis, _ = SURFACES[wall]
    return axis, {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]


def wall_face_rectangle(trap, dimensions_m):
    """A wall trap's face: (normal axis, the plane's coordinate, [(lo, hi) along the width axis, (lo, hi) along the
    height axis], the two in-plane axes). RAISES when centre_m is not on its surface or the face pokes past it."""
    if trap["wall"] not in SURFACES:
        raise RuntimeError(f"(╯°□°)╯ trap {trap['name']}: wall {trap['wall']!r} is not one of {sorted(SURFACES)}")
    axis, far = SURFACES[trap["wall"]]
    plane = dimensions_m[axis] if far else 0.0
    centre = trap["centre_m"]
    if abs(centre[axis] - plane) > ON_SURFACE_M:
        raise RuntimeError(f"(╯°□°)╯ trap {trap['name']}: centre {centre} is not on surface {trap['wall']} "
                           f"({'xyz'[axis]} = {plane})")
    _, (width_axis, height_axis) = surface_axes(trap["wall"])
    spans = [(centre[width_axis] - trap["width_m"] / 2, centre[width_axis] + trap["width_m"] / 2),
             (centre[height_axis] - trap["height_m"] / 2, centre[height_axis] + trap["height_m"] / 2)]
    for (lo, hi), along in zip(spans, (width_axis, height_axis)):
        if lo < -ON_SURFACE_M or hi > dimensions_m[along] + ON_SURFACE_M:
            raise RuntimeError(f"(╯°□°)╯ trap {trap['name']}: its face spans {'xyz'[along]} {lo:.4g}..{hi:.4g}, "
                               f"past its surface (0..{dimensions_m[along]})")
    return axis, plane, spans, (width_axis, height_axis)


def rectangles_overlap(a, b, dimensions_m):
    if a["wall"] != b["wall"]:
        return False
    _, _, spans_a, _ = wall_face_rectangle(a, dimensions_m)
    _, _, spans_b, _ = wall_face_rectangle(b, dimensions_m)
    return all(lo_a < hi_b - ON_SURFACE_M and lo_b < hi_a - ON_SURFACE_M
               for (lo_a, hi_a), (lo_b, hi_b) in zip(spans_a, spans_b))


def trap_face_frame(trap, dimensions_m):
    """A trap's face frame: {"face_centre_m", "normal" (into the room), "along", "up"} and, for a corner prism (a
    triangular prism in a vertical corner, legs along the two walls, its 45-degree face `width_m` wide and `height_m`
    tall, standing on the floor or hanging from the ceiling), "corner_xy", "into_room_xy" (each leg's direction's
    signs), "leg_m", "z_range_m". On a wall "along" = z x normal (to the right of someone in the room facing the trap)
    and "up" = z; on the floor or the ceiling "along" = x and "up" = y."""
    lx, ly, lz = dimensions_m
    if trap["mount"] == "corner":
        i, j = trap["corner"]
        leg = trap["width_m"] / math.sqrt(2)
        cx, cy = (0.0, lx)[i], (0.0, ly)[j]
        sx, sy = (1.0, -1.0)[i], (1.0, -1.0)[j]
        normal = [sx / math.sqrt(2), sy / math.sqrt(2), 0.0]
        z_range = (0.0, trap["height_m"]) if trap["level"] == "floor" else (lz - trap["height_m"], lz)
        return {"corner_xy": (cx, cy), "into_room_xy": (sx, sy), "leg_m": leg, "z_range_m": z_range,
                "face_centre_m": [cx + sx * leg / 2, cy + sy * leg / 2, sum(z_range) / 2],
                "normal": normal, "along": [-normal[1], normal[0], 0.0], "up": [0.0, 0.0, 1.0]}
    axis, far = SURFACES[trap["wall"]]
    wall_face_rectangle(trap, dimensions_m)
    normal = [0.0, 0.0, 0.0]
    normal[axis] = -1.0 if far else 1.0
    if axis == 2:
        along, up = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    else:
        along, up = [-normal[1], normal[0], 0.0], [0.0, 0.0, 1.0]
    return {"face_centre_m": list(trap["centre_m"]), "normal": normal, "along": along, "up": up}


def surface_keep_outs(traps, dimensions_m):
    """{surface: [((u0, u1), (v0, v1))]}: what a trap covers on each room surface, in the surface's (width, height)
    axes (surface_axes): a wall trap's face; a corner prism's footprint on its two walls (leg x its z range) and on
    the floor or the ceiling (the leg x leg square holding its triangle)."""
    lx, ly, lz = dimensions_m
    keep_outs = {wall: [] for wall in SURFACES}
    for trap in traps:
        if trap["mount"] == "wall":
            _, _, spans, _ = wall_face_rectangle(trap, dimensions_m)
            keep_outs[trap["wall"]].append(tuple(spans))
            continue
        frame = trap_face_frame(trap, dimensions_m)
        (cx, cy), (sx, sy), leg, z_range = frame["corner_xy"], frame["into_room_xy"], frame["leg_m"], frame["z_range_m"]
        x_span = tuple(sorted((cx, cx + sx * leg)))
        y_span = tuple(sorted((cy, cy + sy * leg)))
        keep_outs["x0" if cx == 0.0 else "x1"].append((y_span, tuple(z_range)))
        keep_outs["y0" if cy == 0.0 else "y1"].append((x_span, tuple(z_range)))
        keep_outs["z0" if trap["level"] == "floor" else "z1"].append((x_span, y_span))
    return keep_outs


RHO_C_TILE_ITERATIONS = 50


def rho_c_tiles(dimensions_m, fraction, tile_m, traps):
    """THE WALLS' rho c TILING: {surface: [((u0, u1), (v0, v1))]}, the absorbing rectangles of every room surface in
    its (width, height) axes. Each surface is cut into ceil(u / tile_m) x ceil(v / tile_m) equal cells, and each cell
    holds one centred rectangle, the cell scaled by one s per surface. A rectangle that would touch a trap's keep-out
    (surface_keep_outs) is left out, and s is set so the surface's rectangles absorb `fraction` of its WHOLE area: a
    trap never takes absorption from the room.

    Why tiles: N equal cells along an axis put the cells' centres where the area average of cos^2(n pi u / L) over
    them is exactly the wall's (the discrete orthogonality of cosines on the midpoints) for every n < N, so to first
    order in the admittance every mode with fewer than N half-waves along each axis of a surface sees the mean
    admittance `fraction` / (rho c) of that surface, as a uniform wall would (the first-visuals' centred patches
    damped the modes unevenly). A fraction a surface cannot hold RAISES."""
    if not 0.0 < fraction < 1.0 or tile_m <= 0.0:
        raise RuntimeError(f"(╯°□°)╯ rho c tiling: fraction {fraction} (0 < f < 1) and tile {tile_m} m (> 0)")
    keep_outs = surface_keep_outs(traps, dimensions_m)
    tiles = {}
    for wall in SURFACES:
        _, (width_axis, height_axis) = surface_axes(wall)
        u, v = dimensions_m[width_axis], dimensions_m[height_axis]
        nu, nv = math.ceil(u / tile_m - 1e-9), math.ceil(v / tile_m - 1e-9)
        du, dv = u / nu, v / nv
        centres = [((i + 0.5) * du, (j + 0.5) * dv) for i in range(nu) for j in range(nv)]
        scale, kept = math.sqrt(fraction), None
        for _ in range(RHO_C_TILE_ITERATIONS):
            half_u, half_v = scale * du / 2, scale * dv / 2
            now = [(cu, cv) for cu, cv in centres
                   if not any(cu - half_u < hu1 + ON_SURFACE_M and hu0 - ON_SURFACE_M < cu + half_u
                              and cv - half_v < hv1 + ON_SURFACE_M and hv0 - ON_SURFACE_M < cv + half_v
                              for (hu0, hu1), (hv0, hv1) in keep_outs[wall])]
            if not now:
                raise RuntimeError(f"(╯°□°)╯ rho c tiling: surface {wall} keeps no tile around its traps")
            wanted = math.sqrt(fraction * nu * nv / len(now))
            if wanted >= 1.0:
                raise RuntimeError(f"(╯°□°)╯ rho c tiling: surface {wall} cannot absorb {fraction:g} of its area in "
                                   f"{len(now)} of its {nu * nv} cells")
            if now == kept and abs(wanted - scale) < 1e-12:
                break
            scale, kept = max(scale, wanted), now
        else:
            raise RuntimeError(f"(╯°□°)╯ rho c tiling: surface {wall} did not settle in {RHO_C_TILE_ITERATIONS} rounds")
        half_u, half_v = scale * du / 2, scale * dv / 2
        tiles[wall] = [((cu - half_u, cu + half_u), (cv - half_v, cv + half_v)) for cu, cv in kept]
    return tiles


def trap_points(spec):
    """Every trap's mic points in the room frame: {"<trap>/<mic>": [x, y, z]}, each mic given as [normal, along, up]
    in its trap's face frame (trap_face_frame: origin the face centre)."""
    points = {}
    for trap in checked_traps(spec):
        frame = trap_face_frame(trap, spec["dimensions_m"])
        for mic, (n, a, u) in trap.get("mic_points_m", {}).items():
            points[f"{trap['name']}/{mic}"] = [frame["face_centre_m"][k] + n * frame["normal"][k]
                                               + a * frame["along"][k] + u * frame["up"][k] for k in range(3)]
    return points


def sensed_names(spec):
    """What the traps can sense, in order: every velocity_port trap's "<trap>/face", then every trap point."""
    return ([f"{trap['name']}/{TRAP_FACE_SENSED}" for trap in checked_traps(spec) if trap["face"] == "velocity_port"]
            + list(trap_points(spec)))


def where_not_in_air(spec, point):
    """Why a point is not in the room's air (outside or on the walls, in or on a source box or a trap prism), or None
    when it is."""
    x, y, z = point
    margin = IN_AIR_MARGIN_M
    if not all(margin < c < length - margin for c, length in zip(point, spec["dimensions_m"])):
        return "outside the room or on its walls"
    for box in checked_sources(spec):
        depth, width, height = box["size_m"]
        bx, by, bz = in_box_frame(box, [point])[0]
        if abs(bx) <= depth / 2 + margin and abs(by) <= width / 2 + margin and abs(bz) <= height / 2 + margin:
            return f"in or on source box {box['name']}"
    for trap in checked_traps(spec):
        if trap["mount"] != "corner":
            continue
        frame = trap_face_frame(trap, spec["dimensions_m"])
        (cx, cy), (sx, sy) = frame["corner_xy"], frame["into_room_xy"]
        u, w = sx * (x - cx), sy * (y - cy)
        low, high = frame["z_range_m"]
        if (u >= -margin and w >= -margin and u + w <= frame["leg_m"] + margin
                and low - margin <= z <= high + margin):
            return f"in or on trap {trap['name']}'s prism"
    return None


def embedded_points(spec):
    """Every point the room mesh embeds as a node: the seat points, then the trap points; one that is not in the air
    RAISES."""
    seats, mics = seat_points(spec), trap_points(spec)
    if set(seats) & set(mics):
        raise RuntimeError(f"(╯°□°)╯ points named both as seat and trap points: {sorted(set(seats) & set(mics))}")
    points = {**seats, **mics}
    for name, point in points.items():
        why = where_not_in_air(spec, point)
        if why:
            raise RuntimeError(f"(╯°□°)╯ point {name} at {[round(c, 4) for c in point]} is {why}: not in the air")
    return points


def field_plane_points(spec, plane):
    """A field plane of the spec ({"name", "normal_axis": "x" | "y" | "z", "at_m", "step_m"}): a grid over the room's
    whole cross-section at that coordinate, cell centres (step / 2 in from each wall, so none is on a wall) ->
    {"points_m": (row, column, 3) (rows along the second in-plane axis, columns the first: x then y then z order),
    "in_air": (row, column) bool (False inside a box or prism), "axes": (first, second)}. A name the spec does not
    have RAISES."""
    planes = {p["name"]: p for p in spec.get("field_planes", [])}
    if plane not in planes:
        raise RuntimeError(f"(╯°□°)╯ spec {spec.get('name')!r} has no field plane {plane!r} ({sorted(planes)})")
    description = planes[plane]
    normal = "xyz".index(description["normal_axis"])
    first, second = [axis for axis in range(3) if axis != normal]
    step = description["step_m"]
    lengths = spec["dimensions_m"]
    along_first = np.arange(step / 2, lengths[first] - step / 2 + 1e-9, step)
    along_second = np.arange(step / 2, lengths[second] - step / 2 + 1e-9, step)
    points = np.zeros((len(along_second), len(along_first), 3))
    points[..., first] = along_first[None, :]
    points[..., second] = along_second[:, None]
    points[..., normal] = description["at_m"]
    in_air = np.array([[where_not_in_air(spec, point) is None for point in row] for row in points])
    return {"points_m": points, "in_air": in_air, "axes": (first, second)}
