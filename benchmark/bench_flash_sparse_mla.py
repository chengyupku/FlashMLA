# Run with:
# python FlashMLA/benchmark/bench_flash_mla.py --one --target flash_mla_sparse

import argparse
import math
import random

import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata


def do_bench(fn, *args, warmup=10, rep=10, **kwargs):
    """
    Do benchmark for a function.
    """
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize()
    for i in range(rep):
        start_event[i].record()
        fn(*args, **kwargs)
        end_event[i].record()
    torch.cuda.synchronize()

    # Record clocks
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )

    return times.mean().item()


# Minimal FP8 KV-cache quantizer (copied from tests/quant.py)
def _quantize_k_cache_fp8(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    dv: int,
    tile_size: int = 128,
):
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    x = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = x.element_size()

    result = torch.empty(
        (num_blocks, block_size, dv + num_tiles * 4 + input_elem_size * (d - dv)),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    result_k_nope_part = result[..., :dv]
    result_k_scale_factor = result[..., dv : dv + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., dv + num_tiles * 4 :].view(x.dtype)
    result_k_rope_part[:] = x[..., dv:]

    for tile_idx in range(0, num_tiles):
        cur_scale_inv = torch.abs(x[..., tile_idx * tile_size : (tile_idx + 1) * tile_size]).max(dim=-1).values / 448.0
        result_k_scale_factor[:, :, tile_idx] = cur_scale_inv
        cur_scale_inv = cur_scale_inv.unsqueeze(-1)
        cur_quant_nope = (x[..., tile_idx * tile_size : (tile_idx + 1) * tile_size].float() / cur_scale_inv.float()).to(
            torch.float8_e4m3fn
        )
        result_k_nope_part[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = cur_quant_nope

    result = result.view(num_blocks, block_size, 1, -1)
    return result


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q, h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_torch, lse_torch = ref_mla()
    t = do_bench(ref_mla)
    return out_torch, lse_torch, t

@torch.inference_mode()
def run_flash_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def flash_mla():
        return flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dv,
            tile_scheduler_metadata, num_splits, causal=causal,
        )

    out_flash, lse_flash = flash_mla()
    t = do_bench(flash_mla)
    return out_flash, lse_flash, t


@torch.inference_mode()
def run_flash_mla_sparse(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
    dtype,
    *,
    topk: int = 2048,
):
    # Generate indices_in_kvcache: [b, s_q, topk]
    indices_in_kvcache = torch.empty(b, s_q, topk, dtype=torch.int32, device=q.device)
    cache_seqlens_cpu = cache_seqlens.cpu()
    block_table_cpu = block_table.cpu()
    for i in range(b):
        cur_len = int(cache_seqlens_cpu[i].item())
        for j in range(s_q):
            if cur_len > 0:
                sel = torch.randperm(cur_len, device="cpu")[:topk]
            else:
                sel = torch.empty(0, dtype=torch.int64, device="cpu")
            if sel.numel() < topk:
                pad = torch.full((topk - sel.numel(),), -1, dtype=torch.int64, device="cpu")
                sel = torch.cat([sel, pad], dim=0)
            blk_idx = torch.where(sel >= 0, sel // block_size, torch.zeros_like(sel))
            off_idx = torch.where(sel >= 0, sel % block_size, torch.zeros_like(sel))
            phys_blk = block_table_cpu[i, blk_idx.clamp_min(0)]
            merged = (phys_blk * block_size + off_idx).to(torch.int32)
            merged[sel < 0] = -1
            indices_in_kvcache[i, j] = merged.to(q.device)

    # Quantize KV cache to FP8-with-scale format (1 head only)
    blocked_k_quant = _quantize_k_cache_fp8(blocked_k, dv, 128)

    # Get schedule metadata for sparse + FP8
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens,
        s_q * h_q // h_kv,
        h_kv,
        h_q,
        True,
        topk,
    )

    def flash_mla_sparse():
        return flash_mla_with_kvcache(
            q,
            blocked_k_quant,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=False,  # sparse mode requires causal=False
            is_fp8_kvcache=True,
            indices=indices_in_kvcache,
        )

    out_flash, lse_flash = flash_mla_sparse()
    t = do_bench(flash_mla_sparse)
    return out_flash, lse_flash, t


FUNC_TABLE = {
    "torch": run_torch_mla,
    "flash_mla": run_flash_mla,
    "flash_mla_sparse": run_flash_mla_sparse,
}
    
def compare_ab(baseline, target, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    print(f"comparing {baseline} vs {target}: {b=}, {s_q=}, mean_seqlens={cache_seqlens.float().mean()}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {dtype=}")
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    assert baseline in FUNC_TABLE
    assert target in FUNC_TABLE
    baseline_func = FUNC_TABLE[baseline]
    target_func = FUNC_TABLE[target]
    
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    
    out_a, lse_a, perf_a = baseline_func(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype
    )
    out_b, lse_b, perf_b = target_func(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype
    )

    # Skip correctness check when comparing sparse vs dense
    skip_correctness = (target == "flash_mla_sparse") or (baseline == "flash_mla_sparse")
    if not skip_correctness:
        torch.testing.assert_close(out_b.float(), out_a.float(), atol=1e-2, rtol=1e-2), "out"
        if target not in ["flash_infer", "flash_mla_triton"] and baseline not in ["flash_infer", "flash_mla_triton"]:
            # flash_infer has a different lse return value; flash_mla_triton doesn't return lse
            torch.testing.assert_close(lse_b.float(), lse_a.float(), atol=1e-2, rtol=1e-2), "lse"

    # Estimate FLOPS/bytes per target
    def est_flops_bytes(target_name: str):
        if target_name == "flash_mla_sparse":
            flops = s_q * b * h_q * TOPK * (d + dv) * 2
            q_bytes = b * s_q * h_q * d * (torch.finfo(dtype).bits // 8)
            kv_bytes = b * s_q * h_kv * TOPK * 656  # 512 fp8 + 16B scales + 128B rope
            out_bytes = b * s_q * h_q * dv * (torch.finfo(dtype).bits // 8)
            return flops, q_bytes + kv_bytes + out_bytes
        else:
            flops = s_q * total_seqlens * h_q * (d + dv) * 2
            bytes_ = (
                total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv
            ) * (torch.finfo(dtype).bits // 8)
            return flops, bytes_

    FLOPS_a, BYTES_a = est_flops_bytes(baseline)
    FLOPS_b, BYTES_b = est_flops_bytes(target)
    print(
        f"perf {baseline}: {perf_a:.3f} ms, {FLOPS_a / 10 ** 9 / perf_a:.0f} TFLOPS, {BYTES_a / 10 ** 6 / perf_a:.0f} GB/s"
    )
    print(
        f"perf {target}: {perf_b:.3f} ms, {FLOPS_b / 10 ** 9 / perf_b:.0f} TFLOPS, {BYTES_b / 10 ** 6 / perf_b:.0f} GB/s"
    )
    return BYTES_a / 10 ** 6 / perf_a, BYTES_b / 10 ** 6 / perf_b


def compare_a(target, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    print(f"{target}: {b=}, {s_q=}, mean_seqlens={cache_seqlens.float().mean()}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {dtype=}")
    torch.set_default_dtype(dtype)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    assert target in FUNC_TABLE
    target_func = FUNC_TABLE[target]
    
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    
    out_b, lse_b, perf_b = target_func(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype)

    if target == "flash_mla_sparse":
        FLOPS = s_q * b * h_q * TOPK * (d + dv) * 2
        q_bytes = b * s_q * h_q * d * (torch.finfo(dtype).bits // 8)
        kv_bytes = b * s_q * h_kv * TOPK * 656
        out_bytes = b * s_q * h_q * dv * (torch.finfo(dtype).bits // 8)
        bytes = q_bytes + kv_bytes + out_bytes
    else:
        FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
        bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    print(
        f"perf {target}: {perf_b:.3f} ms, {FLOPS / 10 ** 9 / perf_b:.0f} TFLOPS, {bytes / 10 ** 6 / perf_b:.0f} GB/s"
    )
    return bytes / 10 ** 6 / perf_b


available_targets = [
    "torch",
    "flash_mla",
    "flash_mla_sparse",
]

shape_configs = [
    {"b": batch, "s_q": 1, "cache_seqlens": torch.tensor([seqlen + 2 * i for i in range(batch)], dtype=torch.int32, device="cuda"), "h_q": head, "h_kv": 1, "d": 512+64, "dv": 512, "causal": True, "dtype": torch.bfloat16}
    for batch in [128] for seqlen in [1024, 2048, 4096, 8192, 8192*2, 8192*4] for head in [128]
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="torch")
    parser.add_argument("--target", type=str, default="flash_mla")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--topk", type=int, default=2048, help="Top-k for flash_mla_sparse")
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    args = get_args()
    # Expose TOPK global for FLOPS/bytes estimation
    global TOPK
    TOPK = args.topk
    benchmark_type = "all" if args.all else f"{args.baseline}_vs_{args.target}" if args.compare else args.target
    with open(f"{benchmark_type}_perf.csv", "w") as fout:
        fout.write("name,batch,seqlen,head,bw\n")
        for shape in shape_configs:
            if args.all:
                for target in available_targets:
                    perf = compare_a(target, shape["b"], shape["s_q"], shape["cache_seqlens"], shape["h_q"], shape["h_kv"], shape["d"], shape["dv"], shape["causal"], shape["dtype"])
                    fout.write(f'{target},{shape["b"]},{shape["cache_seqlens"].float().mean().cpu().item():.0f},{shape["h_q"]},{perf:.0f}\n')
            elif args.compare:
                perfa, prefb = compare_ab(args.baseline, args.target, shape["b"], shape["s_q"], shape["cache_seqlens"], shape["h_q"], shape["h_kv"], shape["d"], shape["dv"], shape["causal"], shape["dtype"])
                fout.write(f'{args.baseline},{shape["b"]},{shape["cache_seqlens"].float().mean().cpu().item():.0f},{shape["h_q"]},{perfa:.0f}\n')
                fout.write(f'{args.target},{shape["b"]},{shape["cache_seqlens"].float().mean().cpu().item():.0f},{shape["h_q"]},{prefb:.0f}\n')
            elif args.one:
                perf = compare_a(args.target, shape["b"], shape["s_q"], shape["cache_seqlens"], shape["h_q"], shape["h_kv"], shape["d"], shape["dv"], shape["causal"], shape["dtype"])
                fout.write(f'{args.target},{shape["b"]},{shape["cache_seqlens"].float().mean().cpu().item():.0f},{shape["h_q"]},{perf:.0f}\n')
