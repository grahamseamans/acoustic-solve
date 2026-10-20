"""A ROOM SPEC (room.json; acoustic_solve.room.build's docstring is its grammar) read without gmsh: the air, the seat
points and their names, the source box's frame, the traps (their kinds, faces, frames, mic points and names). The
builder writes these and the analysis reads them back, so both import them from here and nothing else spells a seat
point's, a trap point's, a trap group's or a sensed quantity's name.
"""
import math

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
FREE_FIELD_SPECS_FILE = "freefield.json"       # the free-field stage's: {"source", "probes": {probe: [point names]}}
FREE_FIELD_BOX_PROBE = "box"                   # the probe of the source's own free_field_points_m
SEAT_CENTRE_POINT = "x+0y+0"                   # the seat itself, on its ear-height grid
SEAT_HEIGHT_OFFSET_M = 0.2                     # the centre lower and higher
TRAP_FACE_KINDS = ("velocity_port", "rho_c")   # a trap face: a port (any law, post-processed) | a rho c termination
TRAP_FACE_SENSED = "face"                      # "<trap>/face": the face-averaged pressure, a sensed quantity
TRAP_GROUP_PREFIX = "trap:"                    # a trap face's physical group "trap:<name>"; its port is
                                               #   "excitation:trap:<name>" (a port id is "excitation:<group>")
IN_AIR_MARGIN_M = 1e-6                         # an embedded point this close to a boundary is not in the air


def seat_points(spec):
    """Every seat's grid at ear height and its centre SEAT_HEIGHT_OFFSET_M lower and higher:
    {"<seat>/<point>": [x, y, z]}, the grid's points "x<i>y<j>" (signed steps), the others "z-0.2" / "z+0.2"."""
    step, n = spec["seat_grid"]["step_m"], spec["seat_grid"]["half_count"]
    points = {}
    for seat, (x0, y0, z0) in spec["seats_m"].items():
        points.update({f"{seat}/x{i:+d}y{j:+d}": [x0 + i * step, y0 + j * step, z0]
                       for i in range(-n, n + 1) for j in range(-n, n + 1)})
        points[f"{seat}/z-{SEAT_HEIGHT_OFFSET_M:g}"] = [x0, y0, z0 - SEAT_HEIGHT_OFFSET_M]
        points[f"{seat}/z+{SEAT_HEIGHT_OFFSET_M:g}"] = [x0, y0, z0 + SEAT_HEIGHT_OFFSET_M]
    return points


def seat_point_indices(point_names, seat):
    """(the seat's centre index, its ear-height grid's indices) in a list of seat point names."""
    centre = f"{seat}/{SEAT_CENTRE_POINT}"
    if centre not in point_names:
        raise RuntimeError(f"(╯°□°)╯ no seat point {centre!r} among {len(point_names)} points")
    return point_names.index(centre), [i for i, name in enumerate(point_names) if name.startswith(f"{seat}/x")]


def seats_of(point_names):
    """The seats named in a list of seat point names, sorted."""
    return sorted({name.split("/")[0] for name in point_names})


def yaw_of(source):
    """The box's yaw: its front face toward aim_m."""
    return math.atan2(source["aim_m"][1] - source["centre_m"][1], source["aim_m"][0] - source["centre_m"][0])


def in_box_frame(source, points):
    """Room points -> the box frame (centre at the origin, front face along +x)."""
    yaw = yaw_of(source)
    cx, cy, cz = source["centre_m"]
    c, s = math.cos(yaw), math.sin(yaw)
    return [[(x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c, z - cz] for x, y, z in points]


def the_box(source):
    """What makes two specs' free fields one: the box, its ports and its free-field points (not where it stands)."""
    return {key: source.get(key) for key in ("size_m", "ports", "free_field_points_m")}


def checked_traps(spec):
    """The spec's traps (none: []), each refused loudly unless it is {"name", "face": one of TRAP_FACE_KINDS, "corner",
    "width_m", "height_m", optional "mic_points_m"} with a unique name and mic names that cannot be mistaken for the
    face ("face") or split a sensed name ("/")."""
    traps = spec.get("traps", [])
    names = []
    for number, trap in enumerate(traps):
        name = trap.get("name")
        if not isinstance(name, str) or not name or "/" in name:
            raise RuntimeError(f"(╯°□°)╯ trap {number}: name {name!r} (required: a non-empty string without '/')")
        if trap.get("face") not in TRAP_FACE_KINDS:
            raise RuntimeError(f"(╯°□°)╯ trap {name}: face {trap.get('face')!r} is not one of {TRAP_FACE_KINDS} "
                               f"(required, no default)")
        unknown = set(trap) - {"name", "face", "corner", "width_m", "height_m", "mic_points_m"}
        if unknown:
            raise RuntimeError(f"(╯°□°)╯ trap {name}: unknown keys {sorted(unknown)}")
        for mic in trap.get("mic_points_m", {}):
            if not mic or "/" in mic or mic == TRAP_FACE_SENSED:
                raise RuntimeError(f"(╯°□°)╯ trap {name}: mic name {mic!r} (not empty, no '/', not "
                                   f"{TRAP_FACE_SENSED!r})")
        names.append(name)
    if len(set(names)) != len(names):
        raise RuntimeError(f"(╯°□°)╯ trap names {names} are not unique")
    return traps


def trap_group(trap):
    """The physical group of a trap's face (its port, for a velocity_port face, is "excitation:" + this)."""
    return f"{TRAP_GROUP_PREFIX}{trap['name']}"


def trap_face_frame(trap, dimensions_m):
    """A floor-standing triangular prism in a corner, legs along the two walls, its 45-degree face `width_m` wide and
    `height_m` tall -> {"corner_xy", "into_room_xy" (each leg's direction's signs), "leg_m", "face_centre_m",
    "normal" (into the room), "along" (z x normal: to the right of someone in the room facing the trap), "up"}."""
    lx, ly, _ = dimensions_m
    i, j = trap["corner"]
    leg = trap["width_m"] / math.sqrt(2)
    cx, cy = (0.0, lx)[i], (0.0, ly)[j]
    sx, sy = (1.0, -1.0)[i], (1.0, -1.0)[j]
    normal = [sx / math.sqrt(2), sy / math.sqrt(2), 0.0]
    return {"corner_xy": (cx, cy), "into_room_xy": (sx, sy), "leg_m": leg,
            "face_centre_m": [cx + sx * leg / 2, cy + sy * leg / 2, trap["height_m"] / 2],
            "normal": normal, "along": [-normal[1], normal[0], 0.0], "up": [0.0, 0.0, 1.0]}


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
    """Why a point is not in the room's air (outside or on the walls, in or on the source box or a trap prism), or
    None when it is."""
    x, y, z = point
    margin = IN_AIR_MARGIN_M
    if not all(margin < c < length - margin for c, length in zip(point, spec["dimensions_m"])):
        return "outside the room or on its walls"
    depth, width, height = spec["source"]["size_m"]
    bx, by, bz = in_box_frame(spec["source"], [point])[0]
    if abs(bx) <= depth / 2 + margin and abs(by) <= width / 2 + margin and abs(bz) <= height / 2 + margin:
        return "in or on the source box"
    for trap in checked_traps(spec):
        frame = trap_face_frame(trap, spec["dimensions_m"])
        (cx, cy), (sx, sy) = frame["corner_xy"], frame["into_room_xy"]
        u, w = sx * (x - cx), sy * (y - cy)
        if u >= -margin and w >= -margin and u + w <= frame["leg_m"] + margin and z <= trap["height_m"] + margin:
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
