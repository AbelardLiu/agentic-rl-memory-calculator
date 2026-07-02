#!/usr/bin/env python3
"""CLI entrypoint for one-shot model memory calculations."""

from __future__ import annotations

import argparse
import json

from model_memory_core import (
    calculate,
    format_result_markdown,
    load_calculation_input,
    write_json_file,
    write_text_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one model memory calculation from JSON input.")
    parser.add_argument("--input-json", help="Run one calculation from an inline JSON string.")
    parser.add_argument("--input-file", help="Run one calculation from a JSON file.")
    parser.add_argument("--output", help="Write the full calculation result to this JSON file. Defaults to stdout.")
    parser.add_argument("--summary-output", help="Write a Markdown summary to this file.")
    args = parser.parse_args()

    data = load_calculation_input(args.input_json, args.input_file)
    result = calculate(data)
    if args.output:
        write_json_file(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.summary_output:
        write_text_file(args.summary_output, format_result_markdown(result))


if __name__ == "__main__":
    main()
