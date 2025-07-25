"""Helper functions for generating flight analysis outputs."""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("nav_loop")


def _generate_visualisation(log_csv_path, analysis_dir, timestamp):
    """Generate visualization from log file."""
    log_csv_path = str(log_csv_path)
    analysis_dir = str(analysis_dir)
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        df = pd.read_csv(log_csv_path)
        
        if len(df) == 0:
            logger.warning(f"Log file has no data rows: {log_csv_path}")
            return None
        
        # Detect if this is SLAM navigation data
        is_slam_data = False
        if 'state' in df.columns:
            slam_states = df['state'].astype(str).str.contains('WP|TRACKING|BOOTSTRAP|UNSTABLE', na=False)
            is_slam_data = slam_states.any()
        
        if is_slam_data:
            logger.info("Detected SLAM navigation data - generating SLAM-specific visualizations")
            return _generate_slam_visualization(df, analysis_dir, timestamp)
        else:
            logger.info("Detected reactive navigation data - generating standard visualizations")
            return _generate_reactive_visualization(df, analysis_dir, timestamp)
            
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return None

def _generate_slam_visualization(df, analysis_dir, timestamp):
    """Generate SLAM-specific visualizations."""
    import matplotlib.pyplot as plt
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Trajectory plot
        if all(col in df.columns for col in ['pos_x', 'pos_y']):
            ax1.plot(df['pos_x'], df['pos_y'], 'b-', alpha=0.7, linewidth=2)
            ax1.scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0], color='green', s=100, label='Start')
            ax1.scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], color='red', s=100, label='End')
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.set_title('SLAM Trajectory')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Distance to goal over time
        if 'combination_flow' in df.columns:  # We repurposed this for distance
            ax2.plot(df['frame'], df['combination_flow'], 'r-', linewidth=2)
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Distance to Goal (m)')
            ax2.set_title('Distance to Waypoint Over Time')
            ax2.grid(True, alpha=0.3)
        
        # 3. Navigation states
        if 'state' in df.columns:
            states = df['state'].value_counts()
            ax3.pie(states.values, labels=states.index, autopct='%1.1f%%')
            ax3.set_title('Navigation State Distribution')
        
        # 4. Speed profile
        if 'speed' in df.columns:
            ax4.plot(df['frame'], df['speed'], 'g-', linewidth=2)
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Speed (m/s)')
            ax4.set_title('Speed Profile')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = Path(analysis_dir) / f"slam_navigation_analysis_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SLAM visualization saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate SLAM visualization: {e}")
        return None

def _generate_reactive_visualization(df, analysis_dir, timestamp):
    """Generate reactive navigation visualizations (existing code)."""
    # Your existing reactive navigation visualization code here
    pass


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
