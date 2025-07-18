import sys
if "numpy" in sys.modules:
    del sys.modules["numpy"]
import numpy as np
sys.modules["numpy"] = np
import csv
import math
import pytest
from analysis.summarise_runs import summarise_log


def test_summarise_log_basic(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pos_x", "pos_y", "pos_z", "collided"])
        writer.writeheader()
        writer.writerow({"pos_x": 0, "pos_y": 0, "pos_z": 0, "collided": 0})
        writer.writerow({"pos_x": 1, "pos_y": 0, "pos_z": 0, "collided": 1})
        writer.writerow({"pos_x": 2, "pos_y": 0, "pos_z": 0, "collided": 0})

    frames, collisions, distance = summarise_log(str(path))
    assert frames == 3
    assert collisions == 1
    assert distance == pytest.approx(2.0)


def test_summarise_log_missing_collided(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pos_x", "pos_y", "pos_z"])
        writer.writeheader()
        writer.writerow({"pos_x": 0, "pos_y": 0, "pos_z": 0})
        writer.writerow({"pos_x": 0, "pos_y": 1, "pos_z": 1})

    frames, collisions, distance = summarise_log(str(path))
    assert frames == 2
    assert collisions == 0
    expected = math.sqrt(2)
    assert distance == pytest.approx(expected)
