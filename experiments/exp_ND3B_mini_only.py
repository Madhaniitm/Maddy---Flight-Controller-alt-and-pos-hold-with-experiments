"""
EXP-ND3B-Mini: HITL Re-run — GPT-4o-mini only
===============================================
GPT-4o-mini-only re-run after fixing silent API error swallowing.
All settings identical to ND3 (same prompt, scenarios, N=5, auto-approve).
Outputs: ND3B_mini_runs_*.csv, ND3B_mini_camera_*.csv, etc.

Run:
    export GLOG_minloglevel=3
    /opt/homebrew/bin/python3.11 experiments/exp_ND3B_mini_only.py
"""

from exp_ND3B_gemini_mini_rerun import main

if __name__ == "__main__":
    main(orchestrators=["gpt4o_mini"], prefix="ND3B_mini")
