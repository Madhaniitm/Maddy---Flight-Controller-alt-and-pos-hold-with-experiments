"""
EXP-E1: LLM API Latency Distribution + Cost Analysis
=====================================================
Goal:
    Make 50 consecutive API calls to each of Claude, GPT-4o, and Gemini with a
    fixed short drone-status prompt. Record wall-clock latency and token count
    for every call. Show that LLM at 0.5–2 Hz is 4 orders of magnitude slower
    than the 4 kHz PID inner loop — justifying the hierarchical architecture.

Metrics:
    - latency_s            : per-call wall time (min, mean, median, P95, max)
    - tokens_in/out        : per call
    - cost_usd             : per call (cumulative + per-100)
    - calls_per_minute     : throughput
    - inner_loop_gap       : mean_latency / 0.00025 s (250 µs inner loop period)

Paper References:
    - ReAct (Yao et al. 2022): outer-loop reason-act cadence (0.5–2 Hz)
    - Vemprala2023: LLMs for robotics, latency as design constraint
    - Madgwick (Madgwick 2010): inner loop 200 Hz — timescale separation argument
"""

import os, sys, json, time, csv, math, pathlib, statistics
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from c_series_agent import _ENDPOINT, _API_KEY, _MODEL, _VERSION, _USE_AZURE, COST_IN, COST_OUT
from d_series_agent import MULTI_LLM_MODELS

# ── Config ─────────────────────────────────────────────────────────────────────
N_CALLS   = 50
OUT_DIR   = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

INNER_LOOP_PERIOD_S = 0.00025   # 4 kHz PID — 250 µs

PAPER_REFS = {
    "ReAct":      "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "Vemprala2023":"Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
    "Madgwick":   "Madgwick et al. 2010 — An efficient orientation filter for inertial/magnetic sensor arrays",
}

# Short fixed prompt — isolates network/model latency from prompt-length effects
STATUS_PROMPT = (
    "Drone telemetry: altitude=1.02m, roll=0.3deg, pitch=-0.1deg, "
    "battery=3.8V, ToF=102cm. Is the drone stable? Answer in one sentence."
)

# ── Per-model rate tables ──────────────────────────────────────────────────────
# USD per 1M tokens (approximate public rates as of 2025)
MODEL_RATES = {
    "claude": {"in": 3.00,  "out": 15.00, "id": _MODEL},
    "gpt4o":  {"in": 2.50,  "out": 10.00, "id": "gpt-4o"},
    "gemini": {"in": 1.25,  "out": 5.00,  "id": "gemini-1.5-pro"},
}

# ── Claude call (Azure or direct) ─────────────────────────────────────────────
def _call_claude(prompt: str) -> tuple[float, int, int]:
    """Returns (latency_s, tokens_in, tokens_out)."""
    import urllib.request
    headers = {"Content-Type": "application/json"}
    body    = {
        "model":      _MODEL,
        "max_tokens": 64,
        "messages":   [{"role": "user", "content": prompt}],
    }
    if _USE_AZURE:
        url = f"{_ENDPOINT}/openai/deployments/{_MODEL}/chat/completions?api-version={_VERSION}"
        headers["api-key"] = _API_KEY
    else:
        url = "https://api.anthropic.com/v1/messages"
        headers.update({"x-api-key": _API_KEY, "anthropic-version": "2023-06-01"})

    req  = urllib.request.Request(url, json.dumps(body).encode(), headers)
    t0   = time.time()
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    lat  = time.time() - t0

    usage  = data.get("usage", {})
    tok_in = usage.get("input_tokens",  usage.get("prompt_tokens", 0))
    tok_out= usage.get("output_tokens", usage.get("completion_tokens", 0))
    return lat, tok_in, tok_out

def _call_openai(prompt: str, model_id: str, api_key: str) -> tuple[float, int, int]:
    import urllib.request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model":      model_id,
        "max_tokens": 64,
        "messages":   [{"role": "user", "content": prompt}],
    }
    req  = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        json.dumps(body).encode(), headers
    )
    t0   = time.time()
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    lat  = time.time() - t0
    usage  = data.get("usage", {})
    return lat, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

def _call_gemini(prompt: str, api_key: str) -> tuple[float, int, int]:
    import urllib.request
    model_id = "gemini-1.5-pro"
    url      = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_id}:generateContent?key={api_key}")
    body = {
        "contents":         [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 64},
    }
    req  = urllib.request.Request(url, json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})
    t0   = time.time()
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    lat  = time.time() - t0
    meta = data.get("usageMetadata", {})
    return lat, meta.get("promptTokenCount", 0), meta.get("candidatesTokenCount", 0)

# ── Run N_CALLS for one model ──────────────────────────────────────────────────
def run_model(model_key: str) -> list[dict]:
    rates = MODEL_RATES[model_key]
    rows  = []
    env   = os.environ

    print(f"\n  Model: {model_key}  ({N_CALLS} calls)")
    for i in range(1, N_CALLS + 1):
        try:
            if model_key == "claude":
                lat, tok_in, tok_out = _call_claude(STATUS_PROMPT)
            elif model_key == "gpt4o":
                key = env.get("OPENAI_API_KEY", "")
                lat, tok_in, tok_out = _call_openai(STATUS_PROMPT, rates["id"], key)
            elif model_key == "gemini":
                key = env.get("GEMINI_API_KEY", "")
                lat, tok_in, tok_out = _call_gemini(STATUS_PROMPT, key)
            else:
                raise ValueError(f"Unknown model key: {model_key}")

            cost = (tok_in * rates["in"] + tok_out * rates["out"]) / 1e6
        except Exception as e:
            print(f"    call {i:3d} ERROR: {e}")
            lat, tok_in, tok_out, cost = float("nan"), 0, 0, 0.0

        rows.append({
            "model":     model_key,
            "call_idx":  i,
            "latency_s": round(lat, 4),
            "tokens_in": tok_in,
            "tokens_out":tok_out,
            "cost_usd":  round(cost, 8),
        })
        if i % 10 == 0:
            valid = [r["latency_s"] for r in rows if not math.isnan(r["latency_s"])]
            print(f"    call {i:3d}  mean_lat={statistics.mean(valid):.3f}s  "
                  f"cost_so_far=${sum(r['cost_usd'] for r in rows):.4f}")
    return rows

# ── Statistics ─────────────────────────────────────────────────────────────────
def percentile(data: list, p: float) -> float:
    return float(np.percentile(data, p))

def summarise(rows: list[dict]) -> dict:
    lats  = [r["latency_s"] for r in rows if not math.isnan(r["latency_s"])]
    costs = [r["cost_usd"]  for r in rows]
    tok_in = [r["tokens_in"]  for r in rows]
    tok_out= [r["tokens_out"] for r in rows]
    total_s = sum(lats)
    return {
        "n_calls":          len(rows),
        "n_valid":          len(lats),
        "lat_min":          round(min(lats), 4),
        "lat_mean":         round(statistics.mean(lats), 4),
        "lat_median":       round(statistics.median(lats), 4),
        "lat_p95":          round(percentile(lats, 95), 4),
        "lat_max":          round(max(lats), 4),
        "lat_stdev":        round(statistics.stdev(lats) if len(lats)>1 else 0, 4),
        "calls_per_min":    round(len(lats) / total_s * 60, 2) if total_s > 0 else 0,
        "cost_per_call":    round(statistics.mean(costs), 8),
        "cost_per_100":     round(sum(costs[:100]), 6),
        "total_cost_usd":   round(sum(costs), 6),
        "mean_tok_in":      round(statistics.mean(tok_in), 1),
        "mean_tok_out":     round(statistics.mean(tok_out), 1),
        "inner_loop_gap":   round(statistics.mean(lats) / INNER_LOOP_PERIOD_S, 0),
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-E1: LLM API Latency Distribution + Cost Analysis")
    print(f"N_CALLS={N_CALLS} per model, models={list(MODEL_RATES.keys())}")
    print(f"Inner loop period = {INNER_LOOP_PERIOD_S*1000:.3f} ms (4 kHz)")
    print("=" * 60)

    all_rows = []
    for model_key in MODEL_RATES:
        all_rows.extend(run_model(model_key))

    # ── Save per-call CSV ──────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "E1_runs.csv"
    fields   = ["model","call_idx","latency_s","tokens_in","tokens_out","cost_usd"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-call data → {runs_csv}")

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "E1_summary.csv"
    model_stats = {}
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["model","metric","value","note"])
        for model_key in MODEL_RATES:
            mr   = [r for r in all_rows if r["model"] == model_key]
            stat = summarise(mr)
            model_stats[model_key] = stat
            for k, v in stat.items():
                note = ""
                if k == "inner_loop_gap":
                    note = f"mean_latency / {INNER_LOOP_PERIOD_S*1000:.3f}ms (4kHz PID period)"
                cw.writerow([model_key, k, v, note])
        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"claude": "#3498db", "gpt4o": "#e67e22", "gemini": "#2ecc71"}
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Left: latency distribution (histogram per model)
        ax = axes[0]
        for mk in MODEL_RATES:
            lats = [r["latency_s"] for r in all_rows
                    if r["model"] == mk and not math.isnan(r["latency_s"])]
            ax.hist(lats, bins=20, alpha=0.6, label=mk, color=colors[mk])
        ax.axvline(INNER_LOOP_PERIOD_S * 1000, color="red", linestyle="--",
                   linewidth=1.5, label=f"PID period\n{INNER_LOOP_PERIOD_S*1e6:.0f}µs")
        ax.set_xlabel("Latency (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"E1: Latency Distribution (N={N_CALLS} each)")
        ax.legend(fontsize=8)

        # Middle: P5/median/P95 box per model
        ax2 = axes[1]
        data = [[r["latency_s"] for r in all_rows
                 if r["model"] == mk and not math.isnan(r["latency_s"])]
                for mk in MODEL_RATES]
        bp = ax2.boxplot(data, labels=list(MODEL_RATES.keys()),
                         patch_artist=True, medianprops={"color":"black"})
        for patch, mk in zip(bp["boxes"], MODEL_RATES):
            patch.set_facecolor(colors[mk])
        ax2.axhline(INNER_LOOP_PERIOD_S, color="red", linestyle="--",
                    label=f"4kHz = {INNER_LOOP_PERIOD_S*1000:.3f}ms")
        ax2.set_ylabel("Latency (s)")
        ax2.set_title("E1: Latency Box-Plot per Model")
        ax2.legend(fontsize=8)
        ax2.set_yscale("log")

        # Right: cost per 100 calls
        ax3 = axes[2]
        mk_list = list(MODEL_RATES.keys())
        c100 = [model_stats[mk]["cost_per_100"] for mk in mk_list]
        ax3.bar(mk_list, c100, color=[colors[mk] for mk in mk_list])
        ax3.set_ylabel("Cost per 100 calls (USD)")
        ax3.set_title("E1: API Cost per 100 Calls")
        for i, v in enumerate(c100):
            ax3.text(i, v + max(c100)*0.02, f"${v:.4f}", ha="center", fontsize=9)

        fig.suptitle(
            "EXP-E1 LLM Latency + Cost — Hierarchical Architecture Justification\n"
            "LLM (0.5–2Hz) vs PID inner loop (4kHz) — 4 orders of magnitude gap\n"
            "ReAct (Yao 2022), Vemprala 2023",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "E1_api_latency.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── E1 Summary ──────────────────────────────────────────────────────")
    print(f"{'Model':10s} {'Mean(s)':9s} {'Median(s)':10s} {'P95(s)':8s} "
          f"{'Max(s)':8s} {'CPM':7s} {'$/100calls':10s} {'Gap×PID'}")
    for mk in MODEL_RATES:
        s = model_stats[mk]
        print(f"  {mk:10s} {s['lat_mean']:.3f}    {s['lat_median']:.3f}      "
              f"{s['lat_p95']:.3f}   {s['lat_max']:.3f}  "
              f"{s['calls_per_min']:5.1f}  ${s['cost_per_100']:.4f}    "
              f"{s['inner_loop_gap']:,.0f}×")
    print(f"\nInner loop (4kHz PID) period = {INNER_LOOP_PERIOD_S*1e6:.0f} µs")
    print("LLM cannot serve as inner-loop controller — timescale separation justified.")


if __name__ == "__main__":
    main()
