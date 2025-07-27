import importlib

import slam_bridge.slam_plotter as sp


def test_main_creates_html(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "analysis").mkdir()

    importlib.reload(sp)

    def fake_plot():
        sp.x_vals[:] = [0, 1]
        sp.y_vals[:] = [0, 0]
        sp.z_vals[:] = [0, 1]
        sp.save_interactive_plot()

    monkeypatch.setattr(sp, "plot_slam_trajectory", fake_plot)
    monkeypatch.setattr(sp.time, "strftime", lambda fmt: "test")

    sp.main()

    files = list((tmp_path / "analysis").glob("slam_traj_*.html"))
    assert len(files) == 1
    assert files[0].read_text().lower().startswith("<html")

