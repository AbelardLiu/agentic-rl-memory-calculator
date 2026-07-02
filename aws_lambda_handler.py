"""AWS Lambda adapter for the model memory calculator.

This file intentionally keeps only HTTP/Lambda glue here. The calculation
logic stays in model_memory_core.py so local web, CLI, GitHub Actions,
and AWS Lambda all use the same implementation.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict

from model_memory_core import calculate, extract_hf_config, format_result_markdown
from model_memory_http_server import INDEX_HTML


CORS_HEADERS = {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type",
}


def _response(status_code: int, body: Any = "", content_type: str = "application/json; charset=utf-8") -> Dict[str, Any]:
    if content_type.startswith("application/json") and not isinstance(body, str):
        payload = json.dumps(body, ensure_ascii=False)
    else:
        payload = str(body)
    return {
        "statusCode": status_code,
        "headers": {
            **CORS_HEADERS,
            "content-type": content_type,
        },
        "body": payload,
        "isBase64Encoded": False,
    }


def _method(event: Dict[str, Any]) -> str:
    request_context = event.get("requestContext") or {}
    http_context = request_context.get("http") or {}
    return str(http_context.get("method") or event.get("httpMethod") or "GET").upper()


def _path(event: Dict[str, Any]) -> str:
    request_context = event.get("requestContext") or {}
    http_context = request_context.get("http") or {}
    return str(event.get("rawPath") or http_context.get("path") or event.get("path") or "/")


def _json_body(event: Dict[str, Any]) -> Dict[str, Any]:
    raw_body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw_body = base64.b64decode(raw_body).decode("utf-8")
    data = json.loads(raw_body or "{}")
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    method = _method(event)
    path = _path(event)

    if method == "OPTIONS":
        return _response(204, "")

    try:
        if method == "GET" and path in {"/", "/index.html"}:
            return _response(200, INDEX_HTML, "text/html; charset=utf-8")

        if method == "GET" and path == "/healthz":
            return _response(200, {"ok": True})

        if method == "POST" and path == "/api/calculate":
            return _response(200, calculate(_json_body(event)))

        if method == "POST" and path == "/api/extract-hf-config":
            data = _json_body(event)
            return _response(200, extract_hf_config(str(data.get("config_json", ""))))

        if method == "POST" and path == "/api/calculate-markdown":
            result = calculate(_json_body(event))
            return _response(200, {"markdown": format_result_markdown(result)})

        return _response(404, {"error": "not found"})
    except Exception as exc:  # noqa: BLE001 - surface input/calculation errors to the browser.
        return _response(400, {"error": str(exc)})
