"""Post-flight reporting utilities for navigation."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import uav.paths as paths
from uav.analysis_helpers import (
    _generate_visualisation,
    _generate_performance,
    _generate_report,
)
from uav.utils import retain_recent_views

logger = logging.getLogger("nav_loop")


def finalise_files(ctx):
    """Generate analysis outputs and clean up artefacts."""
    if ctx is None:
        logger.warning("No context provided for finalization.")
        return

    timestamp = getattr(ctx, "timestamp", None)
    if not timestamp:
        logger.warning("No timestamp found - skipping file finalization")
        return

    logger.info(f"Starting post-flight analysis for timestamp: {timestamp}")

    try:
        base_dir = Path(getattr(ctx, "output_dir", "."))

        log_csv = None
        ctx_log_file = getattr(ctx, "log_file", None)
        if ctx_log_file is not None:
            try:
                log_csv = Path(ctx_log_file.name)
            except Exception:
                logger.warning("Could not determine log file path from ctx.log_file")

        if log_csv is None:
            reactive = base_dir / "flow_logs" / f"reactive_log_{timestamp}.csv"
            slam = base_dir / "flow_logs" / f"slam_log_{timestamp}.csv"
            if reactive.exists():
                log_csv = reactive
            elif slam.exists():
                log_csv = slam
            else:
                logger.error(f"Log file not found: {reactive} or {slam}")
                return

        if not os.path.exists(log_csv):
            logger.error(f"Log file not found: {log_csv}")
            return

        file_size = os.path.getsize(log_csv)
        if file_size < 100:
            logger.warning(
                f"Log file appears empty or corrupt: {log_csv} ({file_size} bytes)"
            )
        logger.info(f"Processing log file: {log_csv} ({file_size} bytes)")

        # DEBUG: Check log file content
        try:
            with open(log_csv, "r") as f:
                lines = f.readlines()
                logger.info(f"Log file has {len(lines)} lines")
                if len(lines) > 0:
                    logger.info(f"First line: {lines[0].strip()}")
                if len(lines) > 1:
                    logger.info(f"Second line: {lines[1].strip()}")
        except Exception as read_error:
            logger.error(f"Could not read log file: {read_error}")
            return

        analysis_dir = base_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = [
                _generate_visualisation(log_csv, analysis_dir, timestamp),
                _generate_performance(log_csv, analysis_dir, timestamp),
                _generate_report(log_csv, analysis_dir, timestamp),
            ]
            if log_csv.name.startswith("slam_log_"):
                script = Path(__file__).resolve().parent.parent / "slam_bridge" / "generate_slam_visualisation.py"
                output_html = analysis_dir / f"slam_trajectory_{timestamp}.html"
                subprocess.run([
                    sys.executable,
                    str(script),
                    str(Path("linux_slam")),
                    "--output",
                    str(output_html),
                ], check=True)
                files.append(str(output_html))
        except subprocess.CalledProcessError as proc_error:
            msg = f"Analysis subprocess failed: {proc_error.stderr}"
            logger.warning(msg)
            logging.getLogger().warning(msg)
            files = []
        except Exception as analysis_error:
            logger.error(f"Analysis generation failed: {analysis_error}")
            files = []

        generated_files = [Path(f) for f in files if f and os.path.exists(f)]

        # Only generate SLAM pose comparison plot in SLAM mode
        nav_mode = getattr(ctx, "nav_mode", None)
        if nav_mode == "slam":
            try:
                from uav import slam_utils

                slam_utils.generate_pose_comparison_plot()
                logger.info("SLAM pose comparison plot generated")
            except Exception as slam_error:
                logger.info(f"SLAM plot generation skipped: {slam_error}")

        if generated_files:
            logger.info("Analysis complete! Generated files:")
            for file_path in generated_files:
                file_size = os.path.getsize(file_path)
                logger.info(f" {file_path} ({file_size} bytes)")
        else:
            logger.warning("No analysis files were successfully generated")

    except Exception as outer_error:
        logger.error(f"Unexpected error in finalise_files: {outer_error}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

    try:
        retain_recent_views(str(analysis_dir), 5)
        logger.info("Old analysis files cleaned up")
    except Exception as cleanup_error:
        logger.error(f"Error retaining recent views: {cleanup_error}")

    try:
        if os.path.exists(paths.STOP_FLAG_PATH):
            os.remove(paths.STOP_FLAG_PATH)
            logger.info("Stop flag file removed")

    except Exception as flag_error:
        logger.error(f"Error removing stop flag file: {flag_error}")

    logger.info("Post-flight analysis finalization complete")
