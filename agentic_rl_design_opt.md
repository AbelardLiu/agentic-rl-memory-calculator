# Agentic RL 显存计算器优化版

这个版本把显存估算的核心计算逻辑从入口脚本里拆出来，让 CLI、HTTP Server、AWS Lambda 和 GitHub Actions 共用同一套实现。

## 文件分层

| 文件 | 职责 |
| --- | --- |
| `model_memory_core.py` | 纯计算逻辑、HuggingFace config 提取、Markdown/JSON 结果格式化 |
| `model_memory_cli.py` | 一次性命令行计算入口，适合本地批处理和 CI |
| `model_memory_http_server.py` | 本地 HTTP 页面和 API server |
| `model_memory_calculator.py` | 兼容旧入口；无输入参数时启动 HTTP server，有输入参数时执行 CLI 计算 |
| `aws_lambda_handler.py` | AWS Lambda Function URL 适配层 |
| `scripts/run_memory_cli.sh` | CLI 调用脚本 |
| `scripts/run_memory_http_server.sh` | HTTP server 调用脚本 |
| `.github/workflows/model-memory-calculator.yml` | GitHub Actions 手动触发计算 |

## 核心逻辑调用

业务代码里直接调用核心模块：

```python
from model_memory_core import calculate

scenario = {
    "params_b": 27,
    "layers": 64,
    "hidden_size": 5120,
    "num_gpus": 4,
    "gpu_memory_gb": 120,
}

result = calculate(scenario)
print(result["training"]["peak_per_gpu_gb"])
```

## CLI 调用

直接运行：

```bash
python model_memory_cli.py \
  --input-file examples/memory_scenario.json \
  --output memory_result.json \
  --summary-output memory_result.md
```

用脚本运行：

```bash
scripts/run_memory_cli.sh examples/memory_scenario.json memory_result.json memory_result.md
```

内联 JSON：

```bash
python model_memory_cli.py \
  --input-json '{"params_b":27,"layers":64,"hidden_size":5120,"num_gpus":4,"gpu_memory_gb":120}'
```

## HTTP Server 调用

启动本地页面和 API：

```bash
python model_memory_http_server.py --host 127.0.0.1 --port 8765
```

或使用脚本：

```bash
HOST=0.0.0.0 PORT=8765 scripts/run_memory_http_server.sh
```

浏览器访问：

```text
http://127.0.0.1:8765
```

API 调用：

```bash
curl -sS http://127.0.0.1:8765/api/calculate \
  -H 'content-type: application/json' \
  -d @examples/memory_scenario.json
```

生成 Markdown：

```bash
curl -sS http://127.0.0.1:8765/api/calculate-markdown \
  -H 'content-type: application/json' \
  -d @examples/memory_scenario.json
```

## GitHub Actions 调用

工作流文件：

```text
.github/workflows/model-memory-calculator.yml
```

手动触发方式：

1. 打开 GitHub 仓库的 `Actions` 页面。
2. 选择 `Model Memory Calculator`。
3. 点击 `Run workflow`。
4. `scenario_json` 可直接粘贴 JSON；留空则读取 `scenario_file`，默认是 `examples/memory_scenario.json`。
5. 执行完成后在 job summary 查看 Markdown 摘要，并从 artifact 下载 `memory_result.json`、`memory_result.md` 和实际输入文件。

命令行触发示例：

```bash
gh workflow run model-memory-calculator.yml \
  -f scenario_file=examples/memory_scenario.json
```

传入内联 JSON：

```bash
gh workflow run model-memory-calculator.yml \
  -f scenario_json='{"params_b":27,"layers":64,"hidden_size":5120,"num_gpus":4,"gpu_memory_gb":120}'
```

## 兼容旧入口

旧命令仍可用：

```bash
python model_memory_calculator.py --host 127.0.0.1 --port 8765
python model_memory_calculator.py --input-file examples/memory_scenario.json
```

新调用建议优先使用 `model_memory_cli.py` 和 `model_memory_http_server.py`，这样入口语义更清晰。
