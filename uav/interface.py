# uav/interface.py
"""GUI helpers used to start and stop the UAV simulation.

The application uses a very small communication mechanism based on flag
files written to the ``flags/`` directory.  The main navigation loop polls
for these files and reacts accordingly:

``nav_mode.flag``
    Created when the user selects a navigation mode and clicks *Launch
    Simulation*.  ``main.py`` reads this to know which mode to run.
``start_nav.flag``
    Indicates that all systems are ready and navigation may begin.
``stop.flag``
    Signals an immediate shutdown of the navigation loop.

The GUI exposes buttons that create these files and provides a minimal STOP
window used during testing.  The :data:`exit_flag` event is also set by the
minimal window so that loops can be terminated gracefully from threads.
"""
import os
import tkinter as tk
from threading import Thread
from threading import Event
from .context import ParamRefs

from .performance import get_cpu_percent, get_memory_info

# Use a multiprocessing.Event to signal when the application should exit
exit_flag = Event()

def launch_control_gui(param_refs, nav_mode="unknown"):
    """Display the main control window.

    Parameters
    ----------
    param_refs : ParamRefs
        Dataclass containing mutable lists used by the navigation loop to expose
        real-time values such as optical flow magnitudes and the current state.
        These values are shown in the GUI and updated periodically.
    nav_mode : str, optional
        Name of the navigation mode to pre-select in the dropdown menu.
        """
    def on_stop():
        """Signal the main loop to terminate by creating a stop flag file."""
        from pathlib import Path
        Path("flags/stop.flag").touch()
        exit_flag.set()
        root.destroy()

    def on_launch_sim():
        """Write selected nav mode to a flag file to trigger simulation startup."""
        from pathlib import Path
        Path("flags/nav_mode.flag").write_text(nav_mode_var.get())

    def on_start_nav():
        """Create the start_nav.flag to trigger navigation."""
        from pathlib import Path
        Path("flags/start_nav.flag").touch()

    def update_labels():
        l_val.set(f"{param_refs['L'][0]:.2f}")  # Flow magnitude label for left sensor
        c_val.set(f"{param_refs['C'][0]:.2f}")  # Flow magnitude label for center sensor
        r_val.set(f"{param_refs['R'][0]:.2f}")  # Flow magnitude label for right sensor
        state_val.set(param_refs['state'][0])  # Current state of the UAV label
        root.after(200, update_labels) # Update every 200ms

    def update_status_lights():
        all_ready = True 
        for name, flag_path in systems:
            if os.path.exists(flag_path):
                status_labels[name].config(fg="green") 
            else:
                status_labels[name].config(fg="red")
                if name in required_systems:
                    all_ready = False
        # Enable the Start Navigation button only if all required flags are green
        if all_ready:
            start_nav_btn.config(state="normal")
        else:
            start_nav_btn.config(state="disabled")
        root.after(500, update_status_lights) # Update every 500ms

    def update_perf():
        cpu = get_cpu_percent()
        mem_mb = get_memory_info().rss / (1024 * 1024)
        perf_val.set(f"CPU: {cpu:.1f}%  MEM: {mem_mb:.1f} MB")
        root.after(1000, update_perf)  # Update every second

    root = tk.Tk() # Create the main application window
    root.title("Hybrid Navigation Simulator") # Set the window title
    root.geometry("340x420") # Set the window size

    l_val = tk.StringVar()  # Flow magnitude for left sensor
    c_val = tk.StringVar()  # Flow magnitude for center sensor
    r_val = tk.StringVar()  # Flow magnitude for right sensor
    state_val = tk.StringVar()  # Current state of the UAV
    perf_val = tk.StringVar()  # CPU & memory usage

    # Set the initial state value
    tk.Label(root, text=f"Navigation Mode: {nav_mode.upper()}", fg="blue", font=("Arial", 10, "bold")).pack(pady=(5, 0))

    # --- Traffic light indicators for all flags ---
    status_frame = tk.Frame(root) # Create a frame for status indicators
    status_frame.pack(pady=10) # Add some padding around the frame

    systems = [
        ("UE4 PID", "flags/ue4_sim.pid"),
        ("AirSim", "flags/airsim_ready.flag"),
        ("SLAM", "flags/slam_ready.flag"),
    ]

    # Navigation mode selection
    nav_modes = ["(select)", "slam", "reactive", "other"]  # List of available navigation modes
    nav_mode_var = tk.StringVar(value=nav_modes[0]) # Default to the first mode

    # List of system names that are required before navigation can start
    def get_required_systems(mode):
        # Always need simulator and AirSim
        base = ["UE4 PID", "AirSim"]
        # Only require SLAM if running in slam mode
        if mode == "slam":
            return base + ["SLAM"]
        return base

    required_systems = get_required_systems(nav_mode_var.get())

    def on_nav_mode_change(*args):
        global required_systems
        required_systems = get_required_systems(nav_mode_var.get())
        update_status_lights()

    nav_mode_var.trace_add("write", on_nav_mode_change)

    status_labels = {} # Dictionary to hold references to status labels
    # Create labels for each system status
    for idx, (name, _) in enumerate(systems):
        tk.Label(status_frame, text=name + ":").grid(row=idx, column=0, sticky='e') # Align label to the right
        lbl = tk.Label(status_frame, text="●", font=("Arial", 18), fg="red") # Create a label for the status light
        lbl.grid(row=idx, column=1, sticky='w') # Align label to the left
        status_labels[name] = lbl # Store the label in the dictionary for later updates


    # Add dropdown and launch button to GUI
    tk.Label(root, text="Select Navigation Mode:").pack(pady=(10, 0)) # Label for dropdown
    nav_mode_menu = tk.OptionMenu(root, nav_mode_var, *nav_modes) # Create a dropdown menu for navigation modes
    nav_mode_menu.pack(pady=(0, 10)) # Add some padding below the dropdown

    # Launch Simulation button
    launch_sim_btn = tk.Button(
        root,
        text="Launch Simulation",
        command=on_launch_sim,
        bg='blue',
        fg='white'
    )
    launch_sim_btn.pack(pady=5)

    # Start Navigation button (initially disabled)
    start_nav_btn = tk.Button(
        root,
        text="Start Navigation",
        command=on_start_nav,
        bg='green',
        fg='white',
        state="disabled"
    )
    start_nav_btn.pack(pady=5)

    # Stop UAV button
    tk.Button(
        root,
        text="Stop UAV",
        command=on_stop,
        bg='red',
        fg='white',
    ).pack(pady=5)

    # Flow magnitude labels
    tk.Label(root, text="Flow Magnitudes").pack(pady=5)

    # Create a frame to hold the flow magnitude labels in a grid layout
    flow_frame = tk.Frame(root)
    flow_frame.pack()
    tk.Label(flow_frame, text="Left:").grid(row=0, column=0, sticky='e')
    tk.Label(flow_frame, textvariable=l_val).grid(row=0, column=1, sticky='w')
    tk.Label(flow_frame, text="Center:").grid(row=1, column=0, sticky='e')
    tk.Label(flow_frame, textvariable=c_val).grid(row=1, column=1, sticky='w')
    tk.Label(flow_frame, text="Right:").grid(row=2, column=0, sticky='e')
    tk.Label(flow_frame, textvariable=r_val).grid(row=2, column=1, sticky='w')

    # Current state label
    tk.Label(root, text="Current State:").pack(pady=(10, 0))
    tk.Label(root, textvariable=state_val).pack()

    tk.Label(root, text="System Usage:").pack(pady=(10, 0))
    tk.Label(root, textvariable=perf_val).pack()

    update_labels()
    update_status_lights()
    update_perf()
    root.mainloop()

def start_gui(param_refs=None, nav_mode="unknown"):
    """Launch the GUI on a daemon thread.
    When ``param_refs`` is provided the full control window is shown;
    otherwise a small STOP window (:func:`gui_exit`) is displayed.  The
    function returns immediately so that the caller can proceed with the
    simulation startup while the GUI runs in the background.
    """
    if param_refs is None:
        gui_exit()
    else:
        launch_control_gui(param_refs, nav_mode)

def gui_exit():
    """Show a small window with a single STOP button.

    Clicking the button sets :data:`exit_flag` which the navigation loops check
    periodically.  This fallback interface is used when the full GUI is not
    started yet but we still want a way to terminate the simulation safely.
    """
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
