from enum import Enum, auto

class NavigationState(Enum):
    """Enumerates high level navigation actions."""

    NONE = auto()
    BRAKE = auto()
    DODGE_LEFT = auto()
    DODGE_RIGHT = auto()
    RESUME = auto()
    BLIND_FORWARD = auto()
    NUDGE = auto()
    RESUME_REINFORCE = auto()
    TIMEOUT_NUDGE = auto()
