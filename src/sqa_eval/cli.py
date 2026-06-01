from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqa_eval.banner import print_banner
from sqa_eval.experiment import Evaluator, Experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sqa-eval",
        description="Speech quality assessment evaluation toolkit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    p_eval = subparsers.add_parser(
        "evaluate",
        help="Score a single audio file.",
    )
    p_eval.add_argument("audio", type=Path, help="Path to the audio file to evaluate.")
    p_eval.add_argument("--ref", type=Path, default=None, help="Path to a clean reference file.")
    p_eval.add_argument(
        "--model",
        default="5metric",
        help="Model alias ('5metric', '22metric') or HF repo ID.",
    )
    p_eval.add_argument("--system", default="default", help="System label for reports.")
    p_eval.add_argument(
        "--output-csv", type=Path, default=None, help="Export per-file scores to CSV."
    )
    p_eval.add_argument(
        "--output-json", type=Path, default=None, help="Export per-file scores to JSON."
    )

    p_dir = subparsers.add_parser(
        "evaluate-dir",
        help="Score all audio files in a directory.",
    )
    p_dir.add_argument("audio_dir", type=Path, help="Directory containing audio files.")
    p_dir.add_argument(
        "--ref-dir", type=Path, default=None, help="Directory with REF_* reference files."
    )
    p_dir.add_argument(
        "--model",
        default="5metric",
        help="Model alias ('5metric', '22metric') or HF repo ID.",
    )
    p_dir.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    p_dir.add_argument(
        "--output-csv", type=Path, default=None, help="Export per-file scores to CSV."
    )
    p_dir.add_argument(
        "--output-json", type=Path, default=None, help="Export per-file scores to JSON."
    )

    p_exp = subparsers.add_parser(
        "experiment",
        help="Compare multiple systems (full report with plots).",
    )
    p_exp.add_argument("name", help="Experiment name (used as output subdirectory name).")
    p_exp.add_argument(
        "base_dir", type=Path, help="Parent directory containing one subdirectory per system."
    )
    p_exp.add_argument(
        "--systems",
        required=True,
        help="Comma-separated system subdirectory names (e.g. v1,v2,v3).",
    )
    p_exp.add_argument(
        "--ref-dir", type=Path, default=None, help="Directory with REF_* reference files."
    )
    p_exp.add_argument(
        "--model",
        default="both",
        help="Model alias ('5metric', '22metric', 'both') or HF repo ID.",
    )
    p_exp.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for reports and plots."
    )

    return parser


def _cmd_evaluate(args: argparse.Namespace) -> int:
    print_banner(model_name=args.model)
    if not args.audio.exists():
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    if args.ref is not None and not args.ref.exists():
        print(f"Error: reference file not found: {args.ref}", file=sys.stderr)
        return 2

    evaluator = Evaluator(model=args.model)
    result = evaluator.evaluate_file(args.audio, ref_path=args.ref, system=args.system)
    results = [result]

    _print_result(result)

    if args.output_csv:
        evaluator.to_csv(results, args.output_csv)
        print(f"Wrote CSV: {args.output_csv.resolve()}")

    if args.output_json:
        evaluator.to_json(results, args.output_json)
        print(f"Wrote JSON: {args.output_json.resolve()}")

    return 0


def _cmd_evaluate_dir(args: argparse.Namespace) -> int:
    print_banner(model_name=args.model)
    if not args.audio_dir.is_dir():
        print(f"Error: audio directory not found: {args.audio_dir}", file=sys.stderr)
        return 2

    evaluator = Evaluator(model=args.model)
    results = evaluator.evaluate_directory(
        args.audio_dir,
        ref_dir=args.ref_dir,
        recursive=args.recursive,
    )

    if not results:
        print(f"No audio files found in {args.audio_dir}", file=sys.stderr)
        return 1

    for r in results:
        _print_result(r)

    if args.output_csv:
        evaluator.to_csv(results, args.output_csv)
        print(f"Wrote CSV: {args.output_csv.resolve()}")

    if args.output_json:
        evaluator.to_json(results, args.output_json)
        print(f"Wrote JSON: {args.output_json.resolve()}")

    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    print_banner(model_name=args.model)
    systems = [s.strip() for s in args.systems.split(",")]

    experiment = Experiment(
        name=args.name,
        base_dir=args.base_dir,
        systems=systems,
        ref_dir=args.ref_dir,
        model=args.model,
        output_dir=args.output_dir,
    )

    results = experiment.run()
    if not results:
        print("No results generated. Check your system directories.", file=sys.stderr)
        return 1

    experiment.report()
    print(f"Experiment complete. Output: {experiment.output_dir.resolve()}")
    return 0


def _print_result(result) -> None:
    print(f"  {result.file_name} ({result.system}) [{result.model_used}]")
    print(f"    common_score:   {result.common_score:.4f}")
    print(f"    extended_score: {result.extended_score:.4f}")
    for name, score in sorted(result.raw_scores.items()):
        print(f"    {name}: {score:.4f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        if args.command == "evaluate-dir":
            return _cmd_evaluate_dir(args)
        if args.command == "experiment":
            return _cmd_experiment(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
