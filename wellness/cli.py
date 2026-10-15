"""Command-line interface for Milestone 1 analysis."""

import argparse
import json
import sys

import nltk

from .errors import WellnessError
from .pipeline import analyze_file, analyze_text


def _download_nltk_data() -> None:
    for package in (
        "stopwords",
        "wordnet",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "vader_lexicon",
    ):
        if not nltk.download(package, quiet=False):
            raise WellnessError(f"Could not download NLTK package: {package}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze employee wellness feedback.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Direct text to analyze.")
    source.add_argument("--file", help="Path to a UTF-8 .txt or .csv file.")
    parser.add_argument("--text-column", help="CSV column containing feedback text.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--download-nltk-data", action="store_true", help="Install required NLTK data.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.download_nltk_data:
            _download_nltk_data()
            print("NLTK data installed.")
            return 0
        if args.text is None and args.file is None:
            raise WellnessError("Provide exactly one of --text or --file.")
        report = (
            analyze_text(args.text)
            if args.text is not None
            else analyze_file(args.file, args.text_column)
        )
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_human_readable())
        return 0
    except WellnessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
