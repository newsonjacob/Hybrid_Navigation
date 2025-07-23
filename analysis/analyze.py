import argparse
from pathlib import Path
from typing import List, Union

from .flight_review import (
    parse_log,
    plot_state_histogram,
    plot_distance_over_time,
)


def parse_args(argv: Union[List[str], None] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis plots")
    parser.add_argument("log", help="CSV log file")
    parser.add_argument(
        "-o", "--outdir", default="analysis", help="Directory to save plots"
    )
    return parser.parse_args(argv)


def main(argv: Union[List[str], None] = None) -> None:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats = parse_log(args.log)
    plot_state_histogram(stats, str(outdir / "state_histogram.html"))
    plot_distance_over_time(args.log, str(outdir / "distance_over_time.html"))


if __name__ == "__main__":
    main()
