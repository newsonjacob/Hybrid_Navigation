"""Post-flight reporting utilities for navigation."""

import logging
import os
import subprocess
from pathlib import Path

from uav.paths import STOP_FLAG_PATH
from uav.analysis_helpers import (
    _generate_visualisation,
    _generate_performance,
    _generate_report,
)

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

    logger.info(f"🎯 Starting post-flight analysis for timestamp: {timestamp}")

    try:
        base_dir = Path(getattr(ctx, "output_dir", "."))
        log_csv = base_dir / "flow_logs" / f"full_log_{timestamp}.csv"

        if not os.path.exists(log_csv):
            logger.error(f"Log file not found: {log_csv}")
            return

        file_size = os.path.getsize(log_csv)
        if file_size < 100:
            logger.warning(
                f"Log file appears empty or corrupt: {log_csv} ({file_size} bytes)"
            )
        logger.info(f"Processing log file: {log_csv} ({file_size} bytes)")

        analysis_dir = base_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = [
                _generate_visualisation(log_csv, analysis_dir, timestamp),
                _generate_performance(log_csv, analysis_dir, timestamp),
                _generate_report(log_csv, analysis_dir, timestamp),
            ]
        except subprocess.CalledProcessError as proc_error:
            logger.error(f"Analysis subprocess failed: {proc_error.stderr}")
            files = []
        except Exception as analysis_error:
            logger.error(f"Analysis generation failed: {analysis_error}")
            files = []

        generated_files = [Path(f) for f in files if f and os.path.exists(f)]

        try:
            from uav import slam_utils
            slam_utils.generate_pose_comparison_plot()
            logger.info("✅ SLAM pose comparison plot generated")
        except Exception as slam_error:
            logger.info(f"SLAM plot generation skipped: {slam_error}")

        if generated_files:
            logger.info("🎯 Analysis complete! Generated files:")
            for file_path in generated_files:
                file_size = os.path.getsize(file_path)
                logger.info(f"  📊 {file_path} ({file_size} bytes)")
        else:
            logger.warning("No analysis files were successfully generated")

    except Exception as outer_error:
        logger.error(f"Unexpected error in finalise_files: {outer_error}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    try:
        from uav.utils import retain_recent_views
        retain_recent_views(str(analysis_dir), 5)
        logger.info("✅ Old analysis files cleaned up")
    except Exception as cleanup_error:
        logger.error(f"Error retaining recent views: {cleanup_error}")

    try:
        if os.path.exists(STOP_FLAG_PATH):
            os.remove(STOP_FLAG_PATH)
            logger.info("✅ Stop flag file removed")
    except Exception as flag_error:
        logger.error(f"Error removing stop flag file: {flag_error}")

    logger.info("🏁 Post-flight analysis finalization complete")
