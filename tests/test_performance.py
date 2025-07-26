import types

from uav import performance


def test_performance_helpers(monkeypatch):
    cpu_val = 42.5
    mem_info = types.SimpleNamespace(rss=123456)

    monkeypatch.setattr(performance.psutil, "cpu_percent", lambda: cpu_val)
    monkeypatch.setattr(
        performance.psutil,
        "Process",
        lambda: types.SimpleNamespace(memory_info=lambda: mem_info),
    )

    assert performance.get_cpu_percent() == cpu_val
    assert performance.get_memory_info() is mem_info
