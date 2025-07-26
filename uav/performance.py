"""CPU and memory usage helpers."""
import psutil


def get_cpu_percent() -> float:
    """Return the current system CPU utilization."""
    return psutil.cpu_percent()


def get_memory_info():
    """Return memory usage for the current process."""
    return psutil.Process().memory_info()
