#!/usr/bin/env python3
"""Pure model memory calculation logic shared by all entrypoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

GB = 1024**3


DTYPE_BYTES = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "nf4": 0.5,
}


@dataclass
class MemoryItem:
    name: str
    global_gb: float
    per_gpu_gb: float
    note: str = ""


def fnum(data: Dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def inum(data: Dict[str, Any], key: str, default: int) -> int:
    return int(round(fnum(data, key, default)))


def bval(data: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def dtype_bytes(dtype: str, default: str = "bf16") -> float:
    return DTYPE_BYTES.get(str(dtype).lower(), DTYPE_BYTES[default])


def gb(value_bytes: float) -> float:
    return value_bytes / GB


def round2(value: float) -> float:
    return round(value + 1e-9, 2)


def safe_div(value: float, parts: int) -> float:
    return value / max(1, parts)


def estimate_lora_params(
    layers: int,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    rank: int,
    target_modules: List[str],
) -> int:
    head_dim = hidden // max(1, heads)
    kv_out = kv_heads * head_dim
    linear_shapes = {
        "q_proj": (hidden, hidden),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (hidden, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    total = 0
    for module in target_modules:
        shape = linear_shapes.get(module.strip())
        if shape:
            in_features, out_features = shape
            total += layers * rank * (in_features + out_features)
    return total


def model_param_count(data: Dict[str, Any]) -> float:
    params_b = fnum(data, "params_b", 27.0)
    return params_b * 1_000_000_000


def base_weight_bytes(data: Dict[str, Any], param_count: float) -> Tuple[float, str]:
    weight_dtype = str(data.get("weight_dtype", "bf16")).lower()
    qlora = bval(data, "qlora", False)
    if qlora:
        bits = fnum(data, "qlora_bits", 4.0)
        overhead = fnum(data, "quant_overhead", 0.18)
        return param_count * bits / 8.0 * (1.0 + overhead), f"QLoRA {bits:g}-bit + {overhead:.0%} scales/metadata"
    return param_count * dtype_bytes(weight_dtype), weight_dtype.upper()


def current_layer_weight_bytes(base_bytes: float, layers: int, fsdp_size: int, use_fsdp: bool) -> float:
    if not use_fsdp:
        return 0.0
    # FSDP all-gathers the current wrapped unit. Use one transformer block as the default unit.
    return base_bytes / max(1, layers)


def tensor_bytes(tokens: float, width: float, dtype: str) -> float:
    return tokens * width * dtype_bytes(dtype)


def make_item(name: str, global_bytes: float, per_gpu_bytes: float, note: str = "") -> MemoryItem:
    return MemoryItem(name, round2(gb(global_bytes)), round2(gb(per_gpu_bytes)), note)


def summarize_items(items: List[MemoryItem]) -> Dict[str, Any]:
    return {
        "items": [item.__dict__ for item in items],
        "total_global_gb": round2(sum(item.global_gb for item in items)),
        "peak_per_gpu_gb": round2(sum(item.per_gpu_gb for item in items)),
    }


def normalize_hf_dtype(value: Any) -> str | None:
    if value is None:
        return None
    name = str(value).lower().replace("torch.", "")
    if name in {"bfloat16", "bf16"}:
        return "bf16"
    if name in {"float16", "fp16", "half"}:
        return "fp16"
    if name in {"float32", "fp32"}:
        return "fp32"
    if name in {"float8", "fp8"}:
        return "fp8"
    return None


def first_config_value(config: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = config.get(key)
        if value not in ("", None):
            return value
    return None


def estimate_decoder_params_from_config(fields: Dict[str, Any], config: Dict[str, Any]) -> float | None:
    direct = first_config_value(config, "num_parameters", "n_params", "total_params", "parameter_count")
    if direct not in ("", None):
        value = float(direct)
        return value / 1_000_000_000 if value > 1_000_000 else value

    required = ["layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size"]
    if any(key not in fields for key in required):
        return None

    layers = int(fields["layers"])
    hidden = int(fields["hidden_size"])
    intermediate = int(fields["intermediate_size"])
    heads = int(fields["num_attention_heads"])
    kv_heads = int(fields["num_key_value_heads"])
    vocab = int(fields["vocab_size"])
    head_dim = hidden // max(1, heads)
    kv_out = kv_heads * head_dim

    qkv_o = hidden * hidden + 2 * hidden * kv_out + hidden * hidden
    mlp = hidden * intermediate * 3
    norms = hidden * 2
    per_layer = qkv_o + mlp + norms
    tie_embeddings = bool(config.get("tie_word_embeddings", False))
    embedding = vocab * hidden
    lm_head = 0 if tie_embeddings else vocab * hidden
    final_norm = hidden
    return (layers * per_layer + embedding + lm_head + final_norm) / 1_000_000_000


def extract_hf_config(raw_config: str) -> Dict[str, Any]:
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError("config.json 内容需要是 JSON object")

    mapping = {
        "layers": ("num_hidden_layers", "n_layer", "num_layers"),
        "hidden_size": ("hidden_size", "n_embd", "d_model"),
        "intermediate_size": ("intermediate_size", "ffn_hidden_size", "n_inner"),
        "num_attention_heads": ("num_attention_heads", "n_head", "num_heads"),
        "num_key_value_heads": ("num_key_value_heads", "n_kv_head", "num_kv_heads"),
        "vocab_size": ("vocab_size",),
    }
    fields: Dict[str, Any] = {}
    for target, keys in mapping.items():
        value = first_config_value(config, *keys)
        if value not in ("", None):
            fields[target] = int(value)

    if "num_key_value_heads" not in fields and "num_attention_heads" in fields:
        fields["num_key_value_heads"] = fields["num_attention_heads"]

    dtype = normalize_hf_dtype(first_config_value(config, "torch_dtype", "dtype"))
    if dtype:
        fields["weight_dtype"] = dtype
        fields["activation_dtype"] = dtype
        fields["kv_cache_dtype"] = dtype

    params_b = estimate_decoder_params_from_config(fields, config)
    if params_b:
        fields["params_b"] = round(params_b, 2)

    max_positions = first_config_value(config, "max_position_embeddings", "seq_length", "max_sequence_length")
    notes = []
    if max_positions:
        notes.append(f"检测到模型最大位置长度 {int(max_positions)}，训练 Seq Len 仍按启动配置手动设置。")
    if "params_b" in fields:
        notes.append("参数量来自 config 直接字段或按 decoder-only Transformer 结构粗估。")
    else:
        notes.append("config 中缺少参数量，且字段不足以粗估参数量。")

    return {"fields": fields, "notes": notes}


def calculate_inference(data: Dict[str, Any]) -> Dict[str, Any]:
    gpus = inum(data, "num_gpus", 4)
    tp = min(max(1, inum(data, "tp_size", gpus)), gpus)
    layers = inum(data, "layers", 64)
    hidden = inum(data, "hidden_size", 5120)
    heads = inum(data, "num_attention_heads", 24)
    kv_heads = inum(data, "num_key_value_heads", 4)
    batch = inum(data, "infer_batch_size", 1)
    prompt_tokens = inum(data, "infer_prompt_tokens", inum(data, "seq_len", 200000))
    gen_tokens = inum(data, "infer_gen_tokens", 1024)
    kv_tokens = inum(data, "kv_cache_tokens", prompt_tokens + gen_tokens)
    act_dtype = str(data.get("activation_dtype", "bf16")).lower()
    kv_dtype = str(data.get("kv_cache_dtype", act_dtype)).lower()
    runtime_gb = fnum(data, "runtime_gb_per_gpu", 4.0)
    param_count = model_param_count(data)
    base_bytes, base_note = base_weight_bytes(data, param_count)
    lora_bytes = lora_adapter_bytes(data)

    head_dim = hidden / max(1, heads)
    kv_cache = batch * kv_tokens * layers * 2 * kv_heads * head_dim * dtype_bytes(kv_dtype)
    hidden_workspace = tensor_bytes(batch * prompt_tokens, hidden, act_dtype) * 2.0

    items = [
        make_item("模型权重", base_bytes, safe_div(base_bytes, tp), f"按 TP={tp} 切分；{base_note}"),
    ]
    if lora_bytes > 0:
        items.append(make_item("LoRA 权重", lora_bytes * gpus, lora_bytes, "推理侧通常每卡加载一份 adapter"))
    items.extend(
        [
            make_item("KV Cache", kv_cache, safe_div(kv_cache, tp), f"{batch} batch, {kv_tokens} tokens, {kv_dtype.upper()}"),
            make_item("推理临时 workspace", hidden_workspace, safe_div(hidden_workspace, tp), "prefill/decode 临时张量粗估"),
            make_item("Runtime / 通信 / 碎片", runtime_gb * GB * gpus, runtime_gb * GB, "CANN/CUDA runtime、通信 buffer 和碎片预留"),
        ]
    )
    result = summarize_items(items)
    result["fit"] = result["peak_per_gpu_gb"] <= fnum(data, "gpu_memory_gb", 120.0)
    return result


def lora_adapter_bytes(data: Dict[str, Any]) -> float:
    if not bval(data, "lora", True):
        return 0.0
    layers = inum(data, "layers", 64)
    hidden = inum(data, "hidden_size", 5120)
    intermediate = inum(data, "intermediate_size", 17408)
    heads = inum(data, "num_attention_heads", 24)
    kv_heads = inum(data, "num_key_value_heads", 4)
    rank = inum(data, "lora_rank", 64)
    targets = str(data.get("lora_targets", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")).split(",")
    params = estimate_lora_params(layers, hidden, intermediate, heads, kv_heads, rank, targets)
    return params * dtype_bytes(str(data.get("adapter_dtype", "bf16")).lower())


def calculate_training(data: Dict[str, Any]) -> Dict[str, Any]:
    gpus = inum(data, "num_gpus", 4)
    gpu_mem = fnum(data, "gpu_memory_gb", 120.0)
    layers = inum(data, "layers", 64)
    hidden = inum(data, "hidden_size", 5120)
    intermediate = inum(data, "intermediate_size", 17408)
    vocab = inum(data, "vocab_size", 151936)
    batch = inum(data, "train_batch_size", 1)
    seq_len = inum(data, "seq_len", 200000)
    fsdp_size = min(max(1, inum(data, "fsdp_size", gpus)), gpus)
    cp_size = min(max(1, inum(data, "cp_size", 1)), gpus)
    use_fsdp = bval(data, "fsdp", True)
    lora = bval(data, "lora", True)
    ce_optimized = bval(data, "ce_optimized", True)
    activation_ckpt = bval(data, "activation_ckpt", True)
    activation_offload = bval(data, "activation_offload", False)
    optimizer_offload = bval(data, "optimizer_offload", False)
    efficient_attention = bval(data, "efficient_attention", True)
    act_dtype = str(data.get("activation_dtype", "bf16")).lower()
    optim_dtype = str(data.get("optimizer_dtype", "fp32")).lower()
    runtime_gb = fnum(data, "runtime_gb_per_gpu", 4.0)
    comm_gb = fnum(data, "comm_gb_per_gpu", 2.0)
    param_count = model_param_count(data)
    base_bytes, base_note = base_weight_bytes(data, param_count)
    lora_bytes = lora_adapter_bytes(data)
    trainable_bytes = lora_bytes if lora else param_count * dtype_bytes(str(data.get("weight_dtype", "bf16")).lower())
    trainable_params = trainable_bytes / dtype_bytes(str(data.get("adapter_dtype" if lora else "weight_dtype", "bf16")).lower())

    if use_fsdp:
        base_resident = safe_div(base_bytes, fsdp_size)
        base_peak_extra = current_layer_weight_bytes(base_bytes, layers, fsdp_size, use_fsdp)
        base_note = f"FSDP={fsdp_size} 常驻分片 + 当前层 all-gather；{base_note}"
    else:
        base_resident = base_bytes
        base_peak_extra = 0.0
        base_note = f"每卡保留全集；{base_note}"

    if use_fsdp:
        grad_per_gpu = safe_div(trainable_bytes, fsdp_size)
        optim_global = trainable_params * 2.0 * dtype_bytes(optim_dtype)
        optim_per_gpu = 0.0 if optimizer_offload else safe_div(optim_global, fsdp_size)
    else:
        grad_per_gpu = trainable_bytes
        optim_global = trainable_params * 2.0 * dtype_bytes(optim_dtype)
        optim_per_gpu = 0.0 if optimizer_offload else optim_global

    local_tokens = batch * math.ceil(seq_len / max(1, cp_size))
    hidden_one = tensor_bytes(local_tokens, hidden, act_dtype)
    intermediate_one = tensor_bytes(local_tokens, intermediate, act_dtype)

    if activation_ckpt:
        stored_act = layers * hidden_one * 1.0
        act_note = "checkpoint 保存 block 输入，反向重算中间激活"
    else:
        stored_act = layers * (2.0 * hidden_one + intermediate_one)
        act_note = "保存主要前向激活，长序列下为高风险项"
    cpu_offload = stored_act if activation_offload else 0.0
    hbm_stored_act = min(stored_act, 2.0 * hidden_one) if activation_offload else stored_act

    if efficient_attention:
        attention_temp = 4.0 * hidden_one
        attn_note = "FlashAttention/SDPA 类路径，按线性 workspace 粗估"
    else:
        heads = inum(data, "num_attention_heads", 24)
        attention_temp = batch * heads * math.ceil(seq_len / max(1, cp_size)) * seq_len * dtype_bytes(act_dtype)
        attn_note = "非 fused attention，包含 score 矩阵，通常不可行"
    mlp_temp = intermediate_one
    ce_chunk = inum(data, "ce_chunk_tokens", 4096)
    ce_tokens = min(local_tokens, ce_chunk) if ce_optimized else local_tokens
    ce_temp = tensor_bytes(ce_tokens, vocab, act_dtype)
    workspace = max(attention_temp, mlp_temp, ce_temp)
    workspace_note = f"max(Attention {gb(attention_temp):.2f}GB, MLP {gb(mlp_temp):.2f}GB, CE {gb(ce_temp):.2f}GB)，这些临时区按复用峰值处理；{attn_note}"

    items = [
        make_item("模型权重常驻", base_resident * gpus, base_resident, base_note),
        make_item("当前层 FSDP all-gather", base_peak_extra * gpus, base_peak_extra, "按一个 transformer block 的临时权重估算"),
    ]
    if lora_bytes > 0:
        adapter_per_gpu = safe_div(lora_bytes, fsdp_size) if use_fsdp else lora_bytes
        items.append(make_item("LoRA adapter 参数", adapter_per_gpu * gpus, adapter_per_gpu, "仅 LoRA 参数参与梯度和优化器状态"))
    items.extend(
        [
            make_item("梯度", grad_per_gpu * gpus, grad_per_gpu, "LoRA 模式只包含 adapter 梯度；全参训练包含全部参数梯度"),
            make_item("优化器状态", optim_per_gpu * gpus, optim_per_gpu, f"Adam m/v，{optim_dtype.upper()}；offload={optimizer_offload}"),
            make_item("保存激活 HBM", hbm_stored_act * gpus, hbm_stored_act, act_note + (f"；CPU offload {gb(cpu_offload):.2f}GB/卡" if activation_offload else "")),
            make_item("可复用临时 workspace", workspace * gpus, workspace, workspace_note),
            make_item("Runtime / 通信 / 碎片", (runtime_gb + comm_gb) * GB * gpus, (runtime_gb + comm_gb) * GB, "Runtime、通信 buffer、allocator 碎片预留"),
        ]
    )
    result = summarize_items(items)
    result["fit"] = result["peak_per_gpu_gb"] <= gpu_mem
    result["cpu_offload_per_gpu_gb"] = round2(gb(cpu_offload))
    result["local_tokens_per_gpu"] = int(local_tokens)
    return result


def calculate_colocated(data: Dict[str, Any], infer: Dict[str, Any], train: Dict[str, Any]) -> Dict[str, Any]:
    gpus = inum(data, "num_gpus", 4)
    gpu_mem = fnum(data, "gpu_memory_gb", 120.0)
    release_kv = bval(data, "release_kv_before_train", True)
    unload_infer = bval(data, "unload_infer_before_train", False)

    infer_items = infer["items"]
    infer_resident = 0.0
    for item in infer_items:
        name = item["name"]
        if unload_infer:
            continue
        if release_kv and name == "KV Cache":
            continue
        if name == "推理临时 workspace":
            continue
        infer_resident += item["per_gpu_gb"]

    train_peak = train["peak_per_gpu_gb"]
    per_gpu = train_peak + infer_resident
    total = per_gpu * gpus
    return {
        "items": [
            {
                "name": "训练峰值",
                "global_gb": round2(train_peak * gpus),
                "per_gpu_gb": round2(train_peak),
                "note": "来自训练场景估算",
            },
            {
                "name": "推理侧未释放常驻项",
                "global_gb": round2(infer_resident * gpus),
                "per_gpu_gb": round2(infer_resident),
                "note": f"release_kv_before_train={release_kv}, unload_infer_before_train={unload_infer}",
            },
        ],
        "total_global_gb": round2(total),
        "peak_per_gpu_gb": round2(per_gpu),
        "fit": per_gpu <= gpu_mem,
    }


def calculate(data: Dict[str, Any]) -> Dict[str, Any]:
    infer = calculate_inference(data)
    train = calculate_training(data)
    colocated = calculate_colocated(data, infer, train)
    return {
        "inference": infer,
        "training": train,
        "colocated": colocated,
        "assumptions": [
            "结果是容量规划估算，不替代框架 profiler 或真实 dry-run。",
            "Attention、MLP、CE logits 这类临时区按可复用 workspace 取最大值，不直接相加。",
            "训练峰值默认不包含 KV Cache；训推共卡场景单独叠加未释放的推理常驻项。",
            "FSDP 默认按一个 transformer block 作为 all-gather 单元估算；实际峰值会受 wrap policy 和 prefetch 影响。",
        ],
    }


def load_calculation_input(input_json: str | None, input_file: str | None) -> Dict[str, Any]:
    if input_json:
        return json.loads(input_json)
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("需要通过 --input-json 或 --input-file 提供计算输入")


def write_json_file(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def format_fit(value: bool) -> str:
    return "可以放下" if value else "显存不足"


def format_items_markdown(title: str, section: Dict[str, Any]) -> str:
    lines = [
        f"## {title}",
        "",
        f"- 总显存估算: {section['total_global_gb']} GB",
        f"- 单卡峰值估算: {section['peak_per_gpu_gb']} GB",
        f"- 结论: {format_fit(bool(section.get('fit')))}",
        "",
        "| 显存项 | 总量 GB | 单卡 GB | 说明 |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in section.get("items", []):
        note = str(item.get("note", "")).replace("\n", " ")
        lines.append(f"| {item['name']} | {item['global_gb']} | {item['per_gpu_gb']} | {note} |")
    return "\n".join(lines)


def format_result_markdown(result: Dict[str, Any]) -> str:
    parts = [
        "# 模型显存计算结果",
        "",
        format_items_markdown("推理场景", result["inference"]),
        "",
        format_items_markdown("训练场景", result["training"]),
        "",
        format_items_markdown("训推共卡场景", result["colocated"]),
        "",
        "## 估算假设",
        "",
    ]
    parts.extend(f"- {item}" for item in result.get("assumptions", []))
    parts.append("")
    return "\n".join(parts)


def write_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


