# uav/interface.py
"""Simple Tkinter GUI utilities for controlling the simulation."""
import tkinter as tk
from threading import Thread
from threading import Event

# Use a multiprocessing.Event to signal when the application should exit
exit_flag = Event()


def launch_control_gui(param_refs):
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

    root = tk.Tk()
    root.title("UAV Controller")
    root.geometry("300x250")

    l_val = tk.StringVar()
    c_val = tk.StringVar()
    r_val = tk.StringVar()
    state_val = tk.StringVar()

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
    tk.Label(flow_frame, textvariable=c_val).grid(
        row=1,
        column=1,
        sticky='w',
    )
    tk.Label(flow_frame, text="Right:").grid(row=2, column=0, sticky='e')
    tk.Label(flow_frame, textvariable=r_val).grid(
        row=2,
        column=1,
        sticky='w',
    )

    tk.Label(root, text="Current State:").pack(pady=(10, 0))
    tk.Label(root, textvariable=state_val).pack()

    update_labels()
    root.mainloop()


def start_gui(param_refs=None):
    """Start the GUI in a background thread."""
    if param_refs is None:
        Thread(
            target=gui_exit,
            daemon=True,
        ).start()
    else:
        Thread(
            target=lambda: launch_control_gui(param_refs),
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
