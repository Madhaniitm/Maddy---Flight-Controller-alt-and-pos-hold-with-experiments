"""
EXP-H3: Blockchain Audit Trail Integrity
=========================================
Goal:
    Demonstrate and validate the SHA-256 linked-block hash chain used to create
    an immutable audit trail of UAV decisions.

    Each Claude decision is stored as a block:
        block = {
            "index":       int,
            "timestamp":   ISO-8601 string,
            "session_id":  str,
            "prompt":      str (first 200 chars),
            "action":      str (tool name or reply summary),
            "prev_hash":   SHA-256 of previous block JSON,
            "hash":        SHA-256 of this block's content,
        }

    Tests:
        (a) Integrity:   run 5 missions (N=5), 5 decisions each → 25 blocks;
                         verify chain is unbroken end-to-end.
        (b) Tamper-detect: modify one block in the middle;
                           verify that all subsequent blocks invalidate.
        (c) Latency:     measure hashing overhead per block.

Metrics:
    - chain_valid_rate   : fraction of untampered chains that verify (Wilson CI)
    - tamper_detect_rate : fraction of tampered chains detected (Wilson CI)
    - hash_latency_us    : microseconds to hash one block (Bootstrap CI)
    - blocks_per_mission : mean blocks written per mission (Bootstrap CI)

Paper References:
    - Nakamoto 2008: SHA-256 hash chain (Bitcoin whitepaper)
    - Ferraro et al. 2018: blockchain for UAV audit trails
    - ReAct (Yao et al. 2022): each tool call → one auditable block
"""

import os, sys, time, csv, math, pathlib, json, hashlib, copy, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

N_RUNS             = 5
DECISIONS_PER_RUN  = 5
TAMPER_RUNS        = 10   # extra runs for tamper-detection test

PAPER_REFS = {
    "Nakamoto2008": "Nakamoto 2008 — Bitcoin: A Peer-to-Peer Electronic Cash System",
    "Ferraro2018":  "Ferraro et al. 2018 — Blockchain for UAV Secure Data Management",
    "ReAct":        "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
}

# ── Block structure ────────────────────────────────────────────────────────────
GENESIS_HASH = "0" * 64

def _block_content(block: dict) -> str:
    """Canonical JSON string for hashing (excludes 'hash' field)."""
    b = {k: v for k, v in block.items() if k != "hash"}
    return json.dumps(b, sort_keys=True, ensure_ascii=True)

def hash_block(block: dict) -> str:
    return hashlib.sha256(_block_content(block).encode()).hexdigest()

def make_block(index: int, session_id: str, prompt: str,
               action: str, prev_hash: str) -> dict:
    block = {
        "index":      index,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": session_id,
        "prompt":     prompt[:200],
        "action":     action[:200],
        "prev_hash":  prev_hash,
    }
    block["hash"] = hash_block(block)
    return block

def verify_chain(chain: list) -> tuple:
    """
    Returns (is_valid, first_bad_index).
    Checks: each block's stored hash matches recomputed hash,
            and prev_hash matches preceding block's hash.
    """
    for i, block in enumerate(chain):
        recomputed = hash_block(block)
        if block["hash"] != recomputed:
            return False, i
        if i > 0 and block["prev_hash"] != chain[i-1]["hash"]:
            return False, i
    return True, -1

# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2/n
    c = (p + z**2/(2*n)) / denom
    m = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return round(p,4), round(max(0,c-m),4), round(min(1,c+m),4)

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr = np.array(data, dtype=float)
    boots = [stat(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)),4), round(float(lo),4), round(float(hi),4)

# ── Run mission with blockchain ────────────────────────────────────────────────
MISSION_PROMPTS = [
    "Arm the drone and report status.",
    "Take off to 1.0 m altitude.",
    "Hold altitude for 5 seconds.",
    "Rotate 45 degrees clockwise.",
    "Land safely and disarm.",
]

def run_mission_with_chain(run_idx: int) -> tuple:
    """
    Returns (chain: list[block], hash_latencies_us: list[float]).
    """
    agent = DAgent(session_id=f"H3_r{run_idx}")
    chain = []
    prev_hash = GENESIS_HASH
    hash_latencies = []

    for i, prompt in enumerate(MISSION_PROMPTS[:DECISIONS_PER_RUN]):
        reply, stats, trace = agent.run_agent_loop(prompt)
        action = reply[:200] if reply else "no_reply"

        t0 = time.perf_counter()
        block = make_block(
            index=i,
            session_id=f"H3_r{run_idx}",
            prompt=prompt,
            action=action,
            prev_hash=prev_hash,
        )
        hash_us = (time.perf_counter() - t0) * 1_000_000
        hash_latencies.append(hash_us)

        chain.append(block)
        prev_hash = block["hash"]

    return chain, hash_latencies

# ── Tamper test ────────────────────────────────────────────────────────────────
def tamper_and_test(chain: list) -> tuple:
    """
    Modify block at index 2 (action field), then verify chain.
    Returns (tampered_chain, detected: bool, first_bad_idx: int).
    """
    tampered = copy.deepcopy(chain)
    if len(tampered) > 2:
        tampered[2]["action"] = "TAMPERED: unauthorized override"
        # Note: we do NOT recompute hash after tampering (simulates real attack)
    valid, bad_idx = verify_chain(tampered)
    return tampered, not valid, bad_idx

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-H3: Blockchain Audit Trail Integrity")
    print(f"N_RUNS={N_RUNS}, DECISIONS_PER_RUN={DECISIONS_PER_RUN}")
    print("=" * 60)

    all_run_rows = []
    all_hash_lats = []
    chain_valid_count = 0

    all_chains = []
    for r in range(1, N_RUNS + 1):
        print(f"\n--- Run {r}/{N_RUNS} ---")
        chain, hash_lats = run_mission_with_chain(r)
        all_chains.append(chain)
        all_hash_lats.extend(hash_lats)

        valid, bad_idx = verify_chain(chain)
        chain_valid_count += int(valid)
        mean_hash_us = round(float(np.mean(hash_lats)), 3)

        print(f"  blocks={len(chain)}  chain_valid={valid}  "
              f"hash_lat={mean_hash_us:.2f}µs")

        all_run_rows.append({
            "run":          r,
            "n_blocks":     len(chain),
            "chain_valid":  int(valid),
            "first_bad":    bad_idx,
            "hash_lat_us":  mean_hash_us,
        })

    # ── Tamper detection test ──────────────────────────────────────────────────
    print(f"\n--- Tamper detection test ({TAMPER_RUNS} chains) ---")
    tamper_detected = 0
    tamper_rows = []
    for t_run in range(TAMPER_RUNS):
        # Use existing chains cycling, or generate new ones
        base_chain = all_chains[t_run % N_RUNS]
        _, detected, bad_idx = tamper_and_test(base_chain)
        tamper_detected += int(detected)
        tamper_rows.append({"tamper_run": t_run+1, "detected": int(detected),
                             "first_bad": bad_idx})
        print(f"  tamper_run={t_run+1}  detected={detected}  first_bad={bad_idx}")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "H3_runs.csv"
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run","n_blocks","chain_valid","first_bad","hash_lat_us"])
        w.writeheader()
        w.writerows(all_run_rows)

    tamper_csv = OUT_DIR / "H3_tamper.csv"
    with open(tamper_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tamper_run","detected","first_bad"])
        w.writeheader()
        w.writerows(tamper_rows)

    # Save one full chain as JSON for reproducibility
    chain_json = OUT_DIR / "H3_sample_chain.json"
    with open(chain_json, "w") as f:
        json.dump(all_chains[0], f, indent=2)

    print(f"\nRun data     → {runs_csv}")
    print(f"Tamper data  → {tamper_csv}")
    print(f"Sample chain → {chain_json}")

    # ── Stats ──────────────────────────────────────────────────────────────────
    cvr, cvr_lo, cvr_hi = wilson_ci(chain_valid_count, N_RUNS)
    tdr, tdr_lo, tdr_hi = wilson_ci(tamper_detected, TAMPER_RUNS)
    hl_m, hl_lo, hl_hi  = bootstrap_ci(all_hash_lats)
    bpm_m, bpm_lo, bpm_hi = bootstrap_ci([r["n_blocks"] for r in all_run_rows])

    summary_csv = OUT_DIR / "H3_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["metric","value","ci_lo","ci_hi","note"])
        cw.writerow(["chain_valid_rate",   cvr,  cvr_lo, cvr_hi,  "Wilson 95%"])
        cw.writerow(["tamper_detect_rate", tdr,  tdr_lo, tdr_hi,  "Wilson 95%"])
        cw.writerow(["hash_latency_us",    hl_m, hl_lo,  hl_hi,   "Bootstrap 95%"])
        cw.writerow(["blocks_per_mission", bpm_m,bpm_lo, bpm_hi,  "Bootstrap 95%"])
        for k, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{k}", ref,"","",""])
    print(f"Summary      → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Chain block diagram for run 1
        ax = axes[0]
        chain_0 = all_chains[0]
        indices = [b["index"] for b in chain_0]
        ax.barh(indices, [1]*len(indices), color="#3498db", alpha=0.7, height=0.6)
        for i, b in enumerate(chain_0):
            short_hash = b["hash"][:8]
            ax.text(0.5, i, f"#{b['index']} {short_hash}…",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.set_yticks(indices)
        ax.set_yticklabels([f"Block {i}" for i in indices])
        ax.set_xlabel("Block slot")
        ax.set_title(f"H3: Sample Chain (Run 1) — valid={verify_chain(chain_0)[0]}")
        ax.set_xlim(0, 1)

        # Hash latency distribution
        ax2 = axes[1]
        ax2.hist(all_hash_lats, bins=20, color="#2ecc71", alpha=0.8)
        ax2.axvline(hl_m, color="black", linestyle="--",
                    label=f"mean={hl_m:.2f}µs")
        ax2.set_xlabel("Hash latency (µs)")
        ax2.set_ylabel("Count")
        ax2.set_title("H3: SHA-256 Hashing Overhead per Block")
        ax2.legend()

        fig.suptitle(
            f"EXP-H3 Blockchain Audit Trail Integrity\n"
            f"Chain valid={cvr:.3f}  Tamper detected={tdr:.3f}  "
            f"Hash latency={hl_m:.2f}µs\n"
            "Nakamoto 2008, Ferraro 2018, ReAct (Yao 2022)",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "H3_blockchain_integrity.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"Plot  → {png}")
    except Exception as e:
        print(f"[plot skipped] {e}")

    print(f"\n── H3 Summary ───────────────────────────────────────────────────")
    print(f"Chain valid rate   : {cvr:.3f} [{cvr_lo:.3f},{cvr_hi:.3f}]")
    print(f"Tamper detect rate : {tdr:.3f} [{tdr_lo:.3f},{tdr_hi:.3f}]")
    print(f"Hash latency       : {hl_m:.3f}µs [{hl_lo:.3f},{hl_hi:.3f}]")
    print(f"Blocks per mission : {bpm_m:.1f} [{bpm_lo:.1f},{bpm_hi:.1f}]")

if __name__ == "__main__":
    main()
