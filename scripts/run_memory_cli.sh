#!/usr/bin/env bash
set -euo pipefail

SCENARIO_FILE="${1:-examples/memory_scenario.json}"
OUTPUT_JSON="${2:-memory_result.json}"
OUTPUT_MD="${3:-memory_result.md}"

python model_memory_cli.py \
  --input-file "$SCENARIO_FILE" \
  --output "$OUTPUT_JSON" \
  --summary-output "$OUTPUT_MD"

echo "JSON result: $OUTPUT_JSON"
echo "Markdown summary: $OUTPUT_MD"
