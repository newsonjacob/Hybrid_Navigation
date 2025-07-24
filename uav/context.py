"""Shared navigation context and thread primitives."""

from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional
from queue import Queue
from threading import Thread, Event


@dataclass
class ParamRefs:
    """Mutable references to share navigation parameters between threads."""

    L: List[float] = field(default_factory=lambda: [0.0])
    C: List[float] = field(default_factory=lambda: [0.0])
    R: List[float] = field(default_factory=lambda: [0.0])
    prev_L: List[float] = field(default_factory=lambda: [0.0])
    prev_C: List[float] = field(default_factory=lambda: [0.0])
    prev_R: List[float] = field(default_factory=lambda: [0.0])
    delta_L: List[float] = field(default_factory=lambda: [0.0])
    delta_C: List[float] = field(default_factory=lambda: [0.0])
    delta_R: List[float] = field(default_factory=lambda: [0.0])
    state: List[Any] = field(default_factory=lambda: [''])
    reset_flag: List[bool] = field(default_factory=lambda: [False])


@dataclass
class NavContext:
    """Container for objects shared across navigation loops."""

    exit_flag: Event
    param_refs: ParamRefs
    tracker: Any
    flow_history: Any
    navigator: Any
    state_history: Deque[Any]
    pos_history: Deque[Any]
    frame_queue: Queue
    video_thread: Thread
    out: Any
    log_file: Any
    log_buffer: List[str] = field(default_factory=list)
    timestamp: str = ''
    start_time: float = 0.0
    fps_list: List[float] = field(default_factory=list)
    fourcc: Any = None
    perception_queue: Optional[Queue] = None
    perception_thread: Optional[Thread] = None
    grace_logged: bool = False
    startup_grace_over: bool = False
    output_dir: str = "."
