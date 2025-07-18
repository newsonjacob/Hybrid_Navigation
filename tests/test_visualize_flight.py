import sys
import types
import pytest
# Replace potential numpy stub from conftest with the real package
if "numpy" in sys.modules and getattr(sys.modules["numpy"], "__file__", None) is None:
    del sys.modules["numpy"]
if "numpy" in sys.modules:
    real_numpy = sys.modules["numpy"]
else:
    import numpy as real_numpy
    sys.modules["numpy"] = real_numpy

# Provide minimal stubs if optional deps are missing
if 'pandas' not in sys.modules:
    sys.modules['pandas'] = types.SimpleNamespace(read_csv=lambda *a, **k: None)

if 'plotly.graph_objects' not in sys.modules:
    class DummyScatter3d:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DummyFigure:
        def __init__(self, data=None):
            self.data = list(data) if data else []
            self.layout = {}
        def update_layout(self, **kwargs):
            self.layout.update(kwargs)

    go_stub = types.SimpleNamespace(Scatter3d=DummyScatter3d, Figure=DummyFigure)
    sys.modules['plotly'] = types.SimpleNamespace(graph_objects=go_stub)
    sys.modules['plotly.graph_objects'] = go_stub

if 'scipy.spatial.transform' not in sys.modules:
    class DummyRotation:
        @staticmethod
        def from_euler(*args, **kwargs):
            class _Rot:
                def as_matrix(self):
                    return [[1,0,0],[0,1,0],[0,0,1]]
            return _Rot()
    transform_stub = types.SimpleNamespace(Rotation=DummyRotation)
    sys.modules['scipy'] = types.SimpleNamespace(spatial=types.SimpleNamespace(transform=transform_stub))
    sys.modules['scipy.spatial'] = types.SimpleNamespace(transform=transform_stub)
    sys.modules['scipy.spatial.transform'] = types.SimpleNamespace(Rotation=DummyRotation)

from analysis import visualise_flight as vis


def test_build_plot_returns_figure(monkeypatch):
    telemetry = vis.np.array([[0, 0, 0], [1, 2, 3]])
    obstacles = [
        {"name": "Box1", "location": [1, 1, 1], "dimensions": [1, 1, 1], "rotation": [0, 0, 0]},
        {"name": "UCX_Skip", "location": [0, 0, 0], "dimensions": [1, 1, 1], "rotation": [0, 0, 0]},
    ]

    # Avoid heavy draw_box computation
    monkeypatch.setattr(vis, "draw_box", lambda *a, **k: ["dummy"], raising=False)

    fig = vis.build_plot(telemetry, obstacles, vis.np.array([0, 0, 0]), scale=1.0)
    assert isinstance(fig, vis.go.Figure)
    # Should contain at least the path trace
    assert len(fig.data) >= 1


def test_find_alignment_marker_success_and_failure():
    obstacles = [
        {"name": "PlayerStart_3", "location": [5, 5, 5]},
        {"name": "Other", "location": [0, 0, 0]}
    ]
    result = vis.find_alignment_marker(obstacles)
    assert vis.np.allclose(result, [5, 5, 5])

    with pytest.raises(ValueError):
        vis.find_alignment_marker([], marker_name="missing")
