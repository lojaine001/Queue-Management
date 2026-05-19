"""
cli.py — Command-line interface for the prediction module.

Provides three subcommands that can be run via:
    python run_prediction.py <command> [options]
or:
    python -m prediction <command> [options]

Subcommands
───────────
  quick      Fast 60-minute Prophet forecast printed as a formatted table.
  pipeline   Full prediction pipeline with metrics and forecast preview.
  actuals    Today's live bucketed entry counts from the database.

All subcommands accept --source REAL|SIM|ALL to filter by data origin.
"""
from __future__ import annotations

import argparse
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quick = subparsers.add_parser("quick", help="Run the quick Prophet forecast for the next 60 minutes.")
    quick.add_argument("--source", type=str, default="REAL", choices=["REAL", "SIM", "ALL"])

    pipeline = subparsers.add_parser("pipeline", help="Run the shared prediction pipeline used by apps.")
    pipeline.add_argument("--source", type=str, default="REAL", choices=["REAL", "SIM", "ALL"])
    pipeline.add_argument("--days", type=int, default=30)
    pipeline.add_argument("--bootstrap", action="store_true")

    actuals = subparsers.add_parser("actuals", help="Print today's live bucketed actuals.")
    actuals.add_argument("--source", type=str, default="REAL", choices=["REAL", "SIM", "ALL"])
    return parser


def _print_quick(result: dict) -> None:
    """Print the quick-forecast result as two formatted console tables.

    Table 1 — Predicted customer entries for each bucket in the next 60 minutes
              (time, predicted count, lower bound, upper bound).
    Table 2 — Estimated queue wait time for each bucket
              (time, entries, estimated wait in minutes).
    Also prints summary wait-time values at +15 and +30 minutes.

    Parameters
    ----------
    result : dict  Return value of run_quick_forecast().
    """
    from .core import BUCKET_MINUTES

    forecast = result["forecast"]
    wait_estimates = result["wait_estimates"]

    print(f"[Quick] Source: {result['source']}")
    print(f"[Quick] Training rows: {len(result['df_combined'])} | real span: {result['real_span_days']:.1f} days")
    print("")
    print("=" * 60)
    print("  PREDICTED CUSTOMER ENTRIES — NEXT 60 MINUTES")
    print("=" * 60)
    print(f"  {'Time':<12} {'Predicted':>10} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for row in forecast.itertuples(index=False):
        print(f"  {row.ds.strftime('%H:%M'):<12} {row.yhat:>10} {row.yhat_lower:>8} {row.yhat_upper:>8}")
    print("=" * 60)
    print("")
    print("  QUEUE WAIT TIME ESTIMATES")
    print(f"  Service rate: {result['service_per_bucket']:.1f} customers/{BUCKET_MINUTES}min  |  Lanes: {result['active_lanes']}")
    print("=" * 60)
    print(f"  {'Time':<12} {'Entries':>10} {'Est. Wait':>12}")
    print("-" * 60)
    for row, wait_row in zip(forecast.itertuples(index=False), wait_estimates):
        wait = float(wait_row["wait_min"])
        print(f"  {row.ds.strftime('%H:%M'):<12} {row.yhat:>10} {wait:>8} min")
    print("=" * 60)
    print(f"\n  Wait @ +15 min: {result['wait_15m']} min  |  Wait @ +30 min: {result['wait_30m']} min")


def _print_pipeline(result: dict) -> None:
    """Print a summary of the full pipeline result to stdout.

    Outputs key metrics (training rows, data span, wait estimates) and a
    10-row preview of the forecast DataFrame.

    Parameters
    ----------
    result : dict  Return value of run_prediction_pipeline().
    """
    print(f"[Pipeline] Source: {result['source']}")
    print(f"[Pipeline] Training rows: {len(result['df_combined'])}")
    print(f"[Pipeline] Real data span: {result['real_span_days']:.1f} days")
    print(f"[Pipeline] Wait @ +15 min: {result['wait_15m']} min")
    print(f"[Pipeline] Wait @ +30 min: {result['wait_30m']} min")
    print("")
    preview = result["forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].head(10).copy()
    preview["ds"] = preview["ds"].dt.strftime("%Y-%m-%d %H:%M")
    print(preview.to_string(index=False))


def _print_actuals(source: str) -> None:
    """Print today's live bucketed arrival counts from the database.

    Parameters
    ----------
    source : str  "REAL", "SIM", or "ALL".
    """
    from .pipeline import load_today_actuals

    df = load_today_actuals(source)
    print(f"[Actuals] Source: {source}")
    if df.empty:
        print("No rows.")
        return
    preview = df[["bucket", "entry_count", "source"]].copy()
    preview["bucket"] = preview["bucket"].dt.strftime("%Y-%m-%d %H:%M")
    print(preview.to_string(index=False))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the prediction CLI.

    Parameters
    ----------
    argv : Sequence[str] | None
        Argument list to parse.  Defaults to sys.argv[1:] when None.

    Returns
    -------
    int
        Exit code: 0 on success, 2 on unknown command.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "quick":
        from .quick import run_quick_forecast

        _print_quick(run_quick_forecast(source=args.source))
        return 0
    if args.command == "pipeline":
        from .pipeline import run_prediction_pipeline

        _print_pipeline(
            run_prediction_pipeline(
                source=args.source,
                days=args.days,
                use_bootstrap=args.bootstrap,
            )
        )
        return 0
    if args.command == "actuals":
        _print_actuals(args.source)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2
