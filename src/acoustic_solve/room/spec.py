"""A ROOM SPEC (room.json; acoustic_solve.room.build's docstring is its grammar) read without gmsh: the air, the seat
points and their names, the source box's frame. The builder writes these and the analysis reads them back, so both
import them from here and nothing else spells a seat point's name.
"""
import math

SOUND_SPEED_M_PER_S = 343.0
DENSITY_KG_PER_M3 = 1.21
POINTS_FILE = "seat_points.json"               # the room stage's: {"seat_points_m": {name: [x, y, z]}, mesh facts}
FREE_FIELD_SPECS_FILE = "freefield.json"       # the free-field stage's: {"source", "probes": {probe: [point names]}}
FREE_FIELD_BOX_PROBE = "box"                   # the probe of the source's own free_field_points_m
SEAT_CENTRE_POINT = "x+0y+0"                   # the seat itself, on its ear-height grid
SEAT_HEIGHT_OFFSET_M = 0.2                     # the centre lower and higher


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
