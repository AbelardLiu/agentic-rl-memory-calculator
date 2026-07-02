#!/usr/bin/env python3
"""HTTP server entrypoint for the model memory calculator."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from model_memory_core import calculate, extract_hf_config, format_result_markdown


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>模型显存计算器</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --line: #d8dee8;
      --text: #18202b;
      --muted: #667085;
      --accent: #2563eb;
      --ok: #0f766e;
      --bad: #b42318;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 480px) 1fr;
      gap: 18px;
      padding: 18px;
      max-width: 1680px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .section-title {
      padding: 12px 14px;
      font-weight: 700;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }
    .form-body {
      display: grid;
      gap: 14px;
      padding: 14px;
    }
    fieldset {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin: 0;
    }
    legend {
      padding: 0 6px;
      font-size: 13px;
      font-weight: 700;
      color: #344054;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    label {
      display: grid;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      min-width: 0;
    }
    input, select {
      width: 100%;
      height: 34px;
      border: 1px solid #c8d0dc;
      border-radius: 6px;
      padding: 0 9px;
      font-size: 13px;
      color: var(--text);
      background: #fff;
    }
    textarea {
      width: 100%;
      min-height: 150px;
      resize: vertical;
      border: 1px solid #c8d0dc;
      border-radius: 6px;
      padding: 9px;
      font-size: 12px;
      line-height: 1.45;
      color: var(--text);
      background: #fff;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      min-height: 34px;
    }
    .check input { width: 16px; height: 16px; }
    .actions {
      display: flex;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: #fbfcfe;
    }
    button {
      height: 36px;
      padding: 0 14px;
      border: 1px solid #1d4ed8;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #fff;
      color: #1d4ed8;
    }
    .inline-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .message {
      font-size: 12px;
      color: var(--muted);
    }
    .results {
      display: grid;
      gap: 14px;
      padding: 14px;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .metric .name { font-size: 13px; color: var(--muted); }
    .metric .value { margin-top: 6px; font-size: 24px; font-weight: 800; }
    .metric .status { margin-top: 6px; font-size: 12px; font-weight: 700; }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    h2 {
      margin: 10px 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }
    th {
      background: #f8fafc;
      color: #344054;
      font-size: 12px;
    }
    tr:last-child td { border-bottom: 0; }
    .notes {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
      padding-left: 18px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .summary { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header><h1>模型显存计算器</h1></header>
  <main>
    <section>
      <div class="section-title">输入参数</div>
      <form id="calc-form">
        <div class="form-body">
          <fieldset>
            <legend>从 HuggingFace config.json 提取</legend>
            <label>
              config.json 内容
              <textarea id="hf-config-json" placeholder='{"hidden_size": 5120, "num_hidden_layers": 64, "intermediate_size": 17408, ...}'></textarea>
            </label>
            <div class="inline-actions">
              <button type="button" class="secondary" id="extract-config">提取并更新模型参数</button>
              <span class="message" id="extract-message"></span>
            </div>
          </fieldset>
          <fieldset>
            <legend>模型配置</legend>
            <div class="grid">
              <label>参数量 B<input name="params_b" type="number" step="0.1" value="27"></label>
              <label>层数<input name="layers" type="number" value="64"></label>
              <label>Hidden Size<input name="hidden_size" type="number" value="5120"></label>
              <label>Intermediate Size<input name="intermediate_size" type="number" value="17408"></label>
              <label>Attention Heads<input name="num_attention_heads" type="number" value="24"></label>
              <label>KV Heads<input name="num_key_value_heads" type="number" value="4"></label>
              <label>Vocab Size<input name="vocab_size" type="number" value="151936"></label>
              <label>权重 dtype
                <select name="weight_dtype">
                  <option value="bf16" selected>BF16</option>
                  <option value="fp16">FP16</option>
                  <option value="fp32">FP32</option>
                  <option value="fp8">FP8</option>
                </select>
              </label>
            </div>
          </fieldset>
          <fieldset>
            <legend>启动配置</legend>
            <div class="grid">
              <label>GPU 数<input name="num_gpus" type="number" value="4"></label>
              <label>单卡显存 GB<input name="gpu_memory_gb" type="number" value="120"></label>
              <label>训练 BS<input name="train_batch_size" type="number" value="1"></label>
              <label>训练 Seq Len<input name="seq_len" type="number" value="200000"></label>
              <label>推理 BS<input name="infer_batch_size" type="number" value="1"></label>
              <label>Prompt Tokens<input name="infer_prompt_tokens" type="number" value="200000"></label>
              <label>Gen Tokens<input name="infer_gen_tokens" type="number" value="1024"></label>
              <label>KV Cache Tokens<input name="kv_cache_tokens" type="number" value="201024"></label>
              <label>激活 dtype
                <select name="activation_dtype">
                  <option value="bf16" selected>BF16</option>
                  <option value="fp16">FP16</option>
                  <option value="fp8">FP8</option>
                </select>
              </label>
              <label>KV dtype
                <select name="kv_cache_dtype">
                  <option value="bf16" selected>BF16</option>
                  <option value="fp16">FP16</option>
                  <option value="fp8">FP8</option>
                </select>
              </label>
              <label>Runtime GB/卡<input name="runtime_gb_per_gpu" type="number" step="0.5" value="4"></label>
              <label>通信/碎片 GB/卡<input name="comm_gb_per_gpu" type="number" step="0.5" value="2"></label>
            </div>
          </fieldset>
          <fieldset>
            <legend>切分方案</legend>
            <div class="grid">
              <label class="check"><input name="fsdp" type="checkbox" checked>FSDP</label>
              <label>FSDP Size<input name="fsdp_size" type="number" value="4"></label>
              <label>CP/SP Size<input name="cp_size" type="number" value="1"></label>
              <label>推理 TP Size<input name="tp_size" type="number" value="4"></label>
            </div>
          </fieldset>
          <fieldset>
            <legend>显存优化</legend>
            <div class="grid">
              <label class="check"><input name="lora" type="checkbox" checked>LoRA</label>
              <label>LoRA Rank<input name="lora_rank" type="number" value="64"></label>
              <label style="grid-column: 1 / -1;">LoRA Targets<input name="lora_targets" value="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"></label>
              <label class="check"><input name="qlora" type="checkbox">QLoRA</label>
              <label>QLoRA Bits<input name="qlora_bits" type="number" value="4"></label>
              <label>量化开销比例<input name="quant_overhead" type="number" step="0.01" value="0.18"></label>
              <label>Adapter dtype
                <select name="adapter_dtype">
                  <option value="bf16" selected>BF16</option>
                  <option value="fp16">FP16</option>
                  <option value="fp32">FP32</option>
                </select>
              </label>
              <label class="check"><input name="ce_optimized" type="checkbox" checked>CE/Logits Chunk</label>
              <label>CE Chunk Tokens<input name="ce_chunk_tokens" type="number" value="4096"></label>
              <label class="check"><input name="activation_ckpt" type="checkbox" checked>Activation Ckpt</label>
              <label class="check"><input name="activation_offload" type="checkbox">Activation Offload</label>
              <label class="check"><input name="optimizer_offload" type="checkbox">Optimizer Offload</label>
              <label class="check"><input name="efficient_attention" type="checkbox" checked>Fused Attention</label>
              <label>Optimizer dtype
                <select name="optimizer_dtype">
                  <option value="fp32" selected>FP32</option>
                  <option value="bf16">BF16</option>
                  <option value="fp16">FP16</option>
                </select>
              </label>
              <label class="check"><input name="release_kv_before_train" type="checkbox" checked>训练前释放 KV</label>
              <label class="check"><input name="unload_infer_before_train" type="checkbox">训练前卸载推理</label>
            </div>
          </fieldset>
        </div>
        <div class="actions">
          <button type="submit">计算</button>
          <button type="button" class="secondary" id="qwen-preset">Qwen3.5-27B 默认值</button>
        </div>
      </form>
    </section>
    <section>
      <div class="section-title">计算结果</div>
      <div class="results" id="results"></div>
    </section>
  </main>
  <script>
    const form = document.getElementById('calc-form');
    const results = document.getElementById('results');
    const extractMessage = document.getElementById('extract-message');
    const numericFields = new Set([
      'params_b','layers','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads',
      'vocab_size','num_gpus','gpu_memory_gb','train_batch_size','seq_len','infer_batch_size',
      'infer_prompt_tokens','infer_gen_tokens','kv_cache_tokens','runtime_gb_per_gpu','comm_gb_per_gpu',
      'fsdp_size','cp_size','tp_size','lora_rank','qlora_bits','quant_overhead','ce_chunk_tokens'
    ]);
    const checkboxFields = new Set([
      'fsdp','lora','qlora','ce_optimized','activation_ckpt','activation_offload',
      'optimizer_offload','efficient_attention','release_kv_before_train','unload_infer_before_train'
    ]);

    function collect() {
      const fd = new FormData(form);
      const data = {};
      for (const el of form.elements) {
        if (!el.name) continue;
        if (checkboxFields.has(el.name)) data[el.name] = el.checked;
        else if (numericFields.has(el.name)) data[el.name] = Number(fd.get(el.name));
        else data[el.name] = fd.get(el.name);
      }
      return data;
    }

    function setField(name, value) {
      const el = form.elements[name];
      if (!el || value === undefined || value === null) return;
      if (el.type === 'checkbox') el.checked = Boolean(value);
      else el.value = value;
    }

    async function extractConfig() {
      const rawConfig = document.getElementById('hf-config-json').value.trim();
      if (!rawConfig) {
        extractMessage.innerHTML = '<span class="warn">请先粘贴 config.json 内容</span>';
        return;
      }
      extractMessage.textContent = '解析中...';
      const resp = await fetch('/api/extract-hf-config', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({config_json: rawConfig})
      });
      const data = await resp.json();
      if (!resp.ok) {
        extractMessage.innerHTML = '<span class="bad">' + (data.error || '解析失败') + '</span>';
        return;
      }
      for (const [name, value] of Object.entries(data.fields || {})) {
        setField(name, value);
      }
      const notes = (data.notes || []).join(' ');
      extractMessage.innerHTML = '<span class="ok">已更新 ' + Object.keys(data.fields || {}).length + ' 个字段。</span> ' + notes;
      calculate();
    }

    function statusText(fit) {
      return fit ? '<span class="status ok">可放入单卡预算</span>' : '<span class="status bad">超过单卡预算</span>';
    }

    function section(title, block) {
      const rows = block.items.map(item => `
        <tr>
          <td>${item.name}</td>
          <td>${item.global_gb.toFixed(2)}</td>
          <td>${item.per_gpu_gb.toFixed(2)}</td>
          <td>${item.note || ''}</td>
        </tr>`).join('');
      return `
        <div>
          <h2>${title}</h2>
          <div class="summary">
            <div class="metric"><div class="name">总显存 GB</div><div class="value">${block.total_global_gb.toFixed(2)}</div></div>
            <div class="metric"><div class="name">单卡峰值 GB</div><div class="value">${block.peak_per_gpu_gb.toFixed(2)}</div>${statusText(block.fit)}</div>
            <div class="metric"><div class="name">CPU Offload GB/卡</div><div class="value">${(block.cpu_offload_per_gpu_gb || 0).toFixed(2)}</div></div>
          </div>
          <table>
            <thead><tr><th style="width: 18%;">项目</th><th style="width: 13%;">总 GB</th><th style="width: 13%;">单卡 GB</th><th>说明</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>`;
    }

    async function calculate() {
      results.innerHTML = '<div class="metric"><div class="name">计算中</div><div class="value">...</div></div>';
      const resp = await fetch('/api/calculate', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify(collect())
      });
      if (!resp.ok) {
        results.innerHTML = '<div class="metric"><div class="name">请求失败</div><div class="value bad">' + resp.status + '</div></div>';
        return;
      }
      const data = await resp.json();
      results.innerHTML = [
        section('推理场景', data.inference),
        section('训练场景', data.training),
        section('训推共卡', data.colocated),
        '<ul class="notes">' + data.assumptions.map(x => '<li>' + x + '</li>').join('') + '</ul>'
      ].join('');
    }

    form.addEventListener('submit', ev => {
      ev.preventDefault();
      calculate();
    });
    document.getElementById('qwen-preset').addEventListener('click', () => {
      form.reset();
      calculate();
    });
    document.getElementById('extract-config').addEventListener('click', extractConfig);
    calculate();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MemoryCalculator/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/healthz":
            self.send_json({"ok": True})
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/calculate", "/api/extract-hf-config", "/api/calculate-markdown"}:
            self.send_error(404, "not found")
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8") or "{}")
            if parsed.path == "/api/calculate":
                self.send_json(calculate(data))
            elif parsed.path == "/api/extract-hf-config":
                self.send_json(extract_hf_config(str(data.get("config_json", ""))))
            else:
                self.send_json({"markdown": format_result_markdown(calculate(data))})
        except Exception as exc:  # noqa: BLE001 - return calculation errors to the browser.
            self.send_response(400)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"))

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_text(self, body: str, content_type: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, body: Dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving model memory calculator at http://{host}:{port}")
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model memory calculator HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
