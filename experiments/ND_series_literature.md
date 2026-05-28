# ND-Series Literature References
**Last updated:** 2026-05-28  
**Scope:** Papers directly informing the design and optimisation of ND1, ND2, ND3 experiments — camera-as-tool agentic drone surveillance for indoor room emergency detection.

---

## 1. Prompt Engineering for LLM-Driven Drone Control

### GSCE: A Prompt Framework with Enhanced Reasoning for Reliable LLM-Driven Drone Control
**Citation:** Wang et al. (2025). arXiv:2502.12531  
**URL:** https://arxiv.org/abs/2502.12531  
**Published:** February 2025  

**What it proposes:**  
GSCE = **G**uidelines · **S**kill APIs · **C**onstraints · **E**xamples — a four-section prompt structure that significantly improves task success rate and completeness for LLM-driven drone control versus unstructured prompts. Constraints bound the LLM's output space; examples (few-shot) demonstrate correct behaviour.

**Applied in ND experiments:**  
`ND_SURVEILLANCE_PROMPT` in `nd_series_agent.py` is restructured as GSCE:
- Goal/Guidelines: drone mission statement and what to look for
- State (sensor data): YOLO-World + MediaPipe detections as structured input
- Constraints: explicit classification rules (EMERGENCY / ALERT / CLEAR) with examples of what each means
- Examples: two few-shot examples (one EMERGENCY, one CLEAR) with full structured output

**Why it matters for ND1:**  
ND1 tests whether the LLM calls analyze_scene and correctly interprets the JSON. An unstructured prompt produces inconsistent STATUS: lines and high "UNKNOWN" rate. GSCE reduces this by anchoring output format with examples.

---

## 2. Tool-Augmented Chain-of-Thought Reasoning

### AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving
**Citation:** (2025). arXiv:2505.15298  
**URL:** https://arxiv.org/abs/2505.15298  
**Published:** May 2025  

**What it proposes:**  
Integrates Chain-of-Thought (CoT) reasoning with dynamic tool invocation. The VLM reasons explicitly before calling each tool, verifies the tool result against expectations, and refines conclusions. Two-stage training: SFT + GRPO. Results: **+53.91% reasoning score, +33.54% answer accuracy** on DriveLMM-o1 benchmark.

**Applied in ND experiments:**  
The outer Claude orchestrator loop (`run_nd_agent_loop`) is already ReAct (Reason→Act→Observe per turn — this is the Anthropic tool-use API loop by design). AgentThink's contribution beyond base ReAct is:
1. Explicit `<thinking>` before tool calls (supported via Claude extended thinking flag)
2. Self-verification after tool result ("does this match my expectation?")

**Why it matters for ND1:**  
Confirms ReAct architecture is correct baseline. The ND1 outer prompt can encourage explicit pre-call reasoning: "Before calling analyze_scene, state what you expect to see and why."

---

## 3. Agentic UAV Architectures for Emergency Detection

### Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning
**Citation:** (2025). arXiv:2509.13352  
**URL:** https://arxiv.org/abs/2509.13352  
**Published:** September 2025  

**What it proposes:**  
Five-layer agentic UAV architecture: Perception → Reasoning → Action → Integration → Learning. Uses YOLOv11 for object detection + GPT-4 for reasoning + locally deployed Gemma 3 for fast inference. LLM reasoning layer raised person detection from **75% → 91%** and detection confidence from **0.72 → 0.79** vs pure YOLO. Multi-agent ecosystem connectivity for database querying and third-party system integration.

**Applied in ND experiments:**  
Validates the three-tier pipeline design (MediaPipe → YOLO-World → LLM). The finding that LLM reasoning over vision outputs improves detection accuracy beyond raw YOLO justifies keeping Tier-3 even when Tier-2 alone would be faster. Suggests YOLOv11 as a future Tier-2 upgrade over YOLO-World for emergency detection tasks.

---

### AI-Enhanced Rescue Drone with Multi-Modal Vision and Cognitive Agentic Architecture
**Citation:** MDPI AI (2025), Vol. 6, No. 10, p. 272  
**URL:** https://www.mdpi.com/2673-2688/6/10/272  
**Published:** October 2025  

**What it proposes:**  
Rescue drone system with custom-trained YOLO11 perception module feeding an LLM cognitive core for contextual hazard analysis. Enables full perception–reasoning–action cycle. Incorporates payload delivery for first-aid supplies. Reduces operator cognitive load through prioritised, actionable recommendations.

**Applied in ND experiments:**  
Closest existing paper to the ND-series architecture. Key differences: ND uses open-vocabulary YOLO-World (no custom training), ND uses four LLMs (not just GPT-4), and ND experiments in sim first. The paper's payload delivery module is analogous to ND's drone control tools executing after analyze_scene recommends an action.

**Reference justification:**  
Justifies the ND architecture as novel: this paper uses a one-shot YOLO→LLM pipeline without the tool-call abstraction, without multi-LLM comparison, and without formal ablation of which pipeline tier contributes most to accuracy.

---

## 4. Hierarchical Agentic Framework for Drone Visual Inspection

### A Hierarchical Agentic Framework for Autonomous Drone-Based Visual Inspection
**Citation:** (2025). arXiv:2510.00259  
**URL:** https://arxiv.org/html/2510.00259v1  
**Published:** October 2025  

**What it proposes:**  
Two-tier hierarchy: Head Agent (plans, coordinates) + Worker Agents (each runs a drone via ReActEval). ReActEval = Reason → Act → Evaluate cycle per command. Head agent maintains cross-task session history; workers reset after each task to prevent context bloat. Best result: **90.5% overall accuracy** with o3 + ReActEval.

**Key finding:** "Method effectiveness is not absolute — it is dictated by a clear interaction between method and model capability." Simpler methods outperform ReActEval with weaker models.

**Applied in ND experiments:**  
ND2 and ND3 use the ReAct-equivalent loop (Anthropic tool-use API). The Evaluate step is partially present via `get_sensor_status()` after each command. The context bloat finding supports ND's `max_turns=30` limit and short history in analyze_scene calls.

**Future work:**  
ND could be extended to a two-drone system (one patrol, one investigation) following this hierarchical architecture.

---

## 5. Accuracy-Latency Tradeoffs for Open-Vocabulary Detection

### Real-Time Open-Vocabulary Perception for Mobile Robots on Edge Devices: A Systematic Analysis of the Accuracy-Latency Trade-off
**Citation:** PMC12583037 (2025)  
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12583037/  
**Published:** 2025  

**What it proposes:**  
Systematic comparison of NanoOWL vs YOLO-World for open-vocabulary detection on embedded hardware.

| Pipeline | Latency | Accuracy | Notes |
|----------|---------|----------|-------|
| NanoOWL (FP16) + EfficientViT-SAM-L0 | **9.81ms (102 FPS)** | 84.64% mIoU | Best speed |
| YOLO-World | ~26 FPS (43% slower) | Higher language understanding | Complex prompts |

**Key findings:**
- NanoOWL optimal for high-responsiveness tasks (obstacle tracking)
- YOLO-World optimal when language-rich class descriptions needed
- Aggressive quantisation (FP16) can cause complete segmentation failure in some models
- Optimal balanced pipeline: **47.51 FPS at 84.64% mIoU**

**Applied in ND experiments:**  
**Optimisation applied:** YOLO confidence threshold filter implemented. Low-confidence detections (conf < 0.35) are excluded from the Tier-3 LLM prompt. Rationale: passing low-confidence YOLO detections as "sensor evidence" introduces noise that confuses the LLM's reasoning (same mechanism as CLIP anchoring degrading reasoning in V_clip_ablation). High-confidence-only input reduces input token count → lower latency + cost.

---

## 6. Low-Latency VLM Inference

### LiteVLM: A Low-Latency Vision-Language Model Inference Pipeline for Resource-Constrained Environments
**Citation:** (2025). arXiv:2506.07416  
**URL:** https://arxiv.org/abs/2506.07416  
**Published:** June 2025 (NVIDIA Research)  

**What it proposes:**  
Three-technique pipeline for embedded VLM deployment:
1. **Patch selection** — filter irrelevant camera views before visual encoding
2. **Token selection** — reduce input sequence length to LLM
3. **Speculative decoding** — accelerate token generation

Result: **2.5× end-to-end latency reduction** without accuracy loss; **3.2× with FP8 quantisation**.

**Applied in ND experiments:**  
Directly motivates two ND optimisations:
1. **max_tokens 300 → 200**: Structured output (STATUS/FINDINGS/DRONE_ACTION/URGENCY/CONFIDENCE) fits in ~150 tokens. Reducing max_tokens lowers generation latency and cost.
2. **YOLO confidence filtering**: Equivalent to token selection — passing only high-confidence detections reduces input length to the Tier-3 LLM.

---

## 7. Multi-Step Reasoning via Tool Augmentation for Embodied QA

### Multi-Step Reasoning for Embodied Question Answering via Tool Augmentation
**Citation:** (2025). arXiv:2510.20310  
**URL:** https://arxiv.org/pdf/2510.20310  
**Published:** October 2025  

**What it proposes:**  
Embodied agents navigate 3D environments and answer questions using explicit multi-step tool use. Tools provide intermediate perceptual grounding that the LLM chains into a reasoning sequence. Addresses VLM hallucinations by making reasoning steps verifiable through tool outputs.

**Applied in ND experiments:**  
Supports the ND architecture's core claim: wrapping the full vision pipeline as a single LLM-callable tool (analyze_scene) provides grounded perceptual output that the outer LLM can reason from — rather than asking the LLM to imagine what the camera sees. The tool result is the observation step in the ReAct loop.

---

## 8. Person Detection in UAV Imagery for Rescue

### Fine-Tuned Deep Models for Niche Datasets — People Detection in UAV Building Images to Aid Rescue Operations
**Citation:** ScienceDirect, International Journal of Applied Earth Observation and Geoscience (2025)  
**URL:** https://www.sciencedirect.com/science/article/pii/S1569843225006326  
**Published:** 2025  

**What it proposes:**  
Fine-tuned YOLO variants for UAV-specific person detection in rescue scenarios. Key finding: standard COCO-trained models underperform on UAV imagery (different perspective, scale, occlusion). Domain-specific fine-tuning with rescue imagery significantly improves person detection recall.

**Applied in ND experiments:**  
ND uses YOLO-World (open-vocabulary, no fine-tuning) + MediaPipe (EfficientDet-Lite0, person detection). This paper supports adding MediaPipe as a second person-detection tier — it compensates for YOLO-World's weaker person detection at UAV perspective. The combination of two person detectors (Tier 1.5 + Tier 2) reduces missed-person rate, which is the critical safety failure mode in room surveillance.

---

## 9. YOLO-World: Open-Vocabulary Real-Time Detection

### YOLO-World: Real-Time Open-Vocabulary Object Detection
**Citation:** Cheng et al. (2024). CVPR 2024  
**URL:** https://github.com/ailab-cvc/yolo-world  
**Published:** CVPR 2024; v2.1 weights released February 2025  

**What it proposes:**  
Extends YOLO with vision-language pre-training (CLIP text encoder) for open-vocabulary detection at real-time speed. Detects arbitrary object classes from text prompts without fine-tuning.

**Applied in ND experiments:**  
Core Tier-2 detector. Used for emergency-relevant class detection: person, fire extinguisher, smoke, water, hazardous material, collapsed structure, etc. Open-vocabulary means we can add new emergency classes to the detection query without retraining.

**Note:** YOLO-World v2.1 (Feb 2025) has improved weights — future upgrade for ND pipeline.

---

## 10. GSCE for Multi-LLM Drone Control Benchmark Context

### LLM Benchmarks 2026: GPT-5, Claude 4.7, Gemini 2.5 Pro, Grok 4 Compared
**URL:** https://futureagi.com/blog/llm-benchmarking-compare-2025/  
**Context:** General reference for LLM capability landscape as of 2026  

**Applied in ND experiments:**  
ND1 compares 4 LLMs (claude, gpt4o, gpt4o_mini, gemini) as Tier-3 vision models. This benchmark context places those models in the broader landscape and justifies the model selection as representative of the current frontier (claude-sonnet-4-6, gpt-4o, gpt-4o-mini, gemini-2.5-flash).

---

## Summary Table — Optimisations Applied to ND Code

| Paper | Optimisation Applied | Where | Status |
|-------|---------------------|-------|--------|
| GSCE (arXiv:2502.12531) | Restructured `ND_SURVEILLANCE_PROMPT` as Goal/State/Constraints/Examples + two few-shot examples | `nd_series_agent.py` | ✅ Done |
| LiteVLM (arXiv:2506.07416) | Right-sized max_tokens per task complexity — ND1 single-shot analysis: 1024 (reasoning + 6-field reply ~600 tokens); ND2/ND3 multi-step patrol: 2048. Avoids over-allocation without risking truncation. | `run_orchestrator_loop()` `max_tokens` param | ✅ Done |
| PMC12583037 | `_filter_yolo_by_confidence(min_conf=0.35)` applied before building Tier-3 prompt | `nd_series_agent.py` `execute_tool("analyze_scene")` | ✅ Done |
| AgentThink (arXiv:2505.15298) + GSCE | Confidence elicitation: `Confidence: <0.0–1.0>` in required reply format; `_extract_confidence()` + `confidence` CSV column — applied at **orchestrator** level (no inner LLM) | `nd_series_agent.py` `ND_SYSTEM_PROMPT` + `exp_ND1_camera_as_tool.py` `RUN_FIELDS` | ✅ Done |
| AgentThink (arXiv:2505.15298) | Confirmed ReAct architecture is correct — Anthropic tool-use API IS ReAct | `run_nd_agent_loop()` | ✅ Confirmed |
| Rescue Drone (arXiv:2509.13352) | Validates three-tier pipeline; LLM over YOLO improves person detection 75→91% | Architecture confirmed | ✅ Confirmed |
| ReActEval (arXiv:2510.00259) | max_turns=30 limit, context management between turns | `run_nd_agent_loop()` | ✅ Done |
| Fine-tuned UAV detection (ScienceDirect) | Dual person detector (MediaPipe + YOLO-World) justified for recall | Architecture confirmed | ✅ Confirmed |

---

## Papers Found But Not Yet Applied

| Paper | Reason Deferred |
|-------|----------------|
| LiteVLM patch selection (FP8 quant) | Requires model-level quantisation, not applicable to API calls |
| AgentThink SFT+GRPO training | Requires fine-tuning, out of scope for ND experiments |
| NanoOWL as Tier-1.5 replacement | Would replace MediaPipe; evaluate after ND1 results |
| YOLOv11 upgrade | Would require updating `enhanced_yolo_pipeline.py`; future work |
| Hierarchical multi-drone (arXiv:2510.00259) | Two-drone extension is future work (ND4?) |
