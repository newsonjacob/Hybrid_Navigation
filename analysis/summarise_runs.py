import csv
import math
from typing import Tuple


def summarise_log(csv_path: str) -> Tuple[int, int, float]:
    """Summarise a flight log CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV log file.

    Returns
    -------
    tuple
        (frame_count, collisions, distance_travelled)
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    frames = len(rows)
    collisions = 0
    coords = []
    for row in rows:
        collisions += int(row.get("collided", 0) or 0)
        try:
            coords.append(
                (float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"]))
            )
        except KeyError:
            pass

    if len(coords) >= 2:
        start = coords[0]
        end = coords[-1]
        distance = math.dist(start, end)
    else:
        distance = 0.0

    return frames, collisions, distance