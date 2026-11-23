import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PhishGuard reproducible experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("baseline")
    training = sub.add_parser("lora")
    training.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 trains on all balanced training rows",
    )
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--eval-samples", type=int, default=256)
    sub.add_parser("quantize")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare

        result = prepare(Path("data/raw/emails.csv"), Path("data/processed"))
        Path("reports").mkdir(exist_ok=True)
        Path("reports/data_manifest.json").write_text(json.dumps(result, indent=2))
    elif args.command == "baseline":
        from .baseline import train_baseline

        result = train_baseline()
    elif args.command == "lora":
        if (
            args.max_samples < 0
            or (args.max_samples and args.max_samples < 2)
            or args.epochs < 1
            or args.eval_samples < 1
        ):
            parser.error("Use max-samples=0 or >=2, epochs>=1, eval-samples>=1")
        from .transformer import train_lora

        result = train_lora(
            max_samples=args.max_samples,
            epochs=args.epochs,
            eval_samples=args.eval_samples,
        )
    else:
        from .transformer import quantize

        result = quantize()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
