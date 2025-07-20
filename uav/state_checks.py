"""Helper checks for navigator state."""


def in_grace_period(current_time, navigator) -> bool:
    """Return True if the UAV is within its post-resume grace period."""

    return navigator.just_resumed and current_time < navigator.resume_grace_end_time
