import main


def test_wait_for_nav_trigger_stops_on_stop(monkeypatch, tmp_path):
    start = tmp_path / "start.flag"
    stop = tmp_path / "stop.flag"
    monkeypatch.setattr(main, "START_FLAG_PATH", start, raising=False)
    monkeypatch.setattr(main, "STOP_FLAG_PATH", stop, raising=False)
    monkeypatch.setattr("uav.paths.STOP_FLAG_PATH", stop, raising=False)

    calls = {"sleep": 0}

    def fake_sleep(_):
        calls["sleep"] += 1
        stop.touch()

    monkeypatch.setattr(main.time, "sleep", fake_sleep)

    main.wait_for_nav_trigger()

    assert stop.exists()
    assert calls["sleep"] == 1
