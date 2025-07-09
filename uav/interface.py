# uav/interface.py
"""Simple Tkinter GUI utilities for controlling the simulation."""
import os
import tkinter as tk
from threading import Thread
from threading import Event

# Use a multiprocessing.Event to signal when the application should exit
exit_flag = Event()


def launch_control_gui(param_refs, nav_mode="unknown"):
    """Launch the full control window using mutable parameter refs."""
    def on_stop():
        """Signal the main loop to terminate."""
        exit_flag.set()

    def update_labels():
        l_val.set(f"{param_refs['L'][0]:.2f}")
        c_val.set(f"{param_refs['C'][0]:.2f}")
        r_val.set(f"{param_refs['R'][0]:.2f}")
        state_val.set(param_refs['state'][0])
        root.after(200, update_labels)

    def update_status_lights():
        for name, flag_path in systems:
            if os.path.exists(flag_path):
                status_labels[name].config(fg="green")
            else:
                status_labels[name].config(fg="red")
        root.after(500, update_status_lights)

    root = tk.Tk()
    root.title("UAV Controller")
    root.geometry("340x380")

    l_val = tk.StringVar()
    c_val = tk.StringVar()
    r_val = tk.StringVar()
    state_val = tk.StringVar()

    tk.Label(root, text=f"Navigation Mode: {nav_mode.upper()}", fg="blue", font=("Arial", 12, "bold")).pack(pady=(5, 0))

    # --- Traffic light indicators for all flags ---
    status_frame = tk.Frame(root)
    status_frame.pack(pady=10)

    systems = [
        ("AirSim", "flags/airsim_ready.flag"),
        ("SLAM", "flags/slam_ready.flag"),
        ("Streamer", "flags/streamer_ready.flag"),
        ("Main", "flags/main_ready.flag"),
        ("Start Nav", "flags/start_nav.flag"),
        ("UE4 PID", "flags/ue4_sim.pid"),
        # Add/remove flags as needed for your workflow
    ]
    status_labels = {}

    for idx, (name, _) in enumerate(systems):
        tk.Label(status_frame, text=name + ":").grid(row=idx, column=0, sticky='e')
        lbl = tk.Label(status_frame, text="●", font=("Arial", 18), fg="red")
        lbl.grid(row=idx, column=1, sticky='w')
        status_labels[name] = lbl

    update_status_lights()

    tk.Button(
        root,
        text="Stop UAV",
        command=on_stop,
        bg='red',
        fg='white',
    ).pack(pady=5)

    tk.Label(root, text="Flow Magnitudes").pack(pady=5)

    flow_frame = tk.Frame(root)
    flow_frame.pack()
    tk.Label(flow_frame, text="Left:").grid(row=0, column=0, sticky='e')
    tk.Label(flow_frame, textvariable=l_val).grid(row=0, column=1, sticky='w')
    tk.Label(flow_frame, text="Center:").grid(row=1, column=0, sticky='e')
    tk.Label(flow_frame, textvariable=c_val).grid(row=1, column=1, sticky='w')
    tk.Label(flow_frame, text="Right:").grid(row=2, column=0, sticky='e')
    tk.Label(flow_frame, textvariable=r_val).grid(row=2, column=1, sticky='w')

    tk.Label(root, text="Current State:").pack(pady=(10, 0))
    tk.Label(root, textvariable=state_val).pack()

    update_labels()
    root.mainloop()


def start_gui(param_refs=None, nav_mode="unknown"):
    """Start the GUI in a background thread."""
    if param_refs is None:
        Thread(
            target=gui_exit,
            daemon=True,
        ).start()
    else:
        Thread(
            target=lambda: launch_control_gui(param_refs, nav_mode),
            daemon=True,
        ).start()


def gui_exit():
    """Display a minimal stop button for emergency exit."""
    root = tk.Tk()
    root.title("Stop UAV")
    root.geometry("200x100")
    btn = tk.Button(
        root,
        text="STOP",
        font=("Arial", 20),
        command=exit_flag.set,
    )
    btn.pack(expand=True)
    root.mainloop()
