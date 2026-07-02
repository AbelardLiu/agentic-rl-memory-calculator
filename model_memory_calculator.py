#!/usr/bin/env python3
"""Backward-compatible launcher for the model memory calculator.

New code should import `model_memory_core` for calculation logic, use
`model_memory_cli.py` for batch jobs, or use `model_memory_http_server.py`
for the local web/API server. This file keeps the old all-in-one command
working.
"""

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
from model_memory_http_server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model memory calculator web app or one-shot CLI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--input-json", help="Run one calculation from an inline JSON string.")
    parser.add_argument("--input-file", help="Run one calculation from a JSON file.")
    parser.add_argument("--output", help="Write the full calculation result to this JSON file.")
    parser.add_argument("--summary-output", help="Write a Markdown summary to this file.")
    args = parser.parse_args()

    if args.input_json or args.input_file:
        data = load_calculation_input(args.input_json, args.input_file)
        result = calculate(data)
        if args.output:
            write_json_file(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.summary_output:
            write_text_file(args.summary_output, format_result_markdown(result))
        return

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
