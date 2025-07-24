"""Helper functions for generating flight analysis outputs."""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("nav_loop")


def _generate_visualisation(log_csv: Path, analysis_dir: Path, timestamp: str) -> str:
    """Generate an interactive flight visualisation HTML file."""
    html_output = str(analysis_dir / f"flight_view_{timestamp}.html")
    logger.info(f"Generating flight visualization: {html_output}")
    script = os.path.abspath("analysis/visualise_flight.py")
    subprocess.run([sys.executable, script, html_output, "--log", str(log_csv)], check=True)
    logger.info(f"✅ Flight visualization saved via subprocess: {html_output}")
    return html_output


def _generate_performance(log_csv: Path, analysis_dir: Path, timestamp: str) -> str:
    """Generate performance plots HTML file."""
    perf_output = str(analysis_dir / f"performance_{timestamp}.html")
    logger.info(f"Generating performance plots: {perf_output}")
    script = os.path.abspath("analysis/performance_plots.py")
    subprocess.run([sys.executable, script, str(log_csv), "--output", perf_output], check=True)
    logger.info(f"✅ Performance plots saved: {perf_output}")
    return perf_output


def _generate_report(log_csv: Path, analysis_dir: Path, timestamp: str):
    """Generate a detailed flight report if the analysis script is available."""
    report_path = str(analysis_dir / f"flight_report_{timestamp}.html")
    analyse_script = os.path.abspath("analysis/analyse.py")
    if not os.path.exists(analyse_script):
        logger.warning("analyse.py not found - skipping flight report")
        return None
    logger.info(f"Generating flight report: {report_path}")
    subprocess.run([
        sys.executable,
        analyse_script,
        str(log_csv),
        "-o",
        report_path,
        "--log-timestamp",
        timestamp,
    ], check=True, capture_output=True, text=True, cwd=os.getcwd(), timeout=60)
    if os.path.exists(report_path):
        logger.info(f"✅ Flight report saved: {report_path}")
        trajectory_path = analysis_dir / f"trajectory_flight_report_{timestamp}.html"
        if os.path.exists(trajectory_path):
            logger.info(f"✅ 3D Trajectory saved: {trajectory_path}")
        else:
            logger.warning(f"3D trajectory file missing: {trajectory_path}")
    else:
        logger.warning(f"Flight report file missing: {report_path}")
    return report_path
