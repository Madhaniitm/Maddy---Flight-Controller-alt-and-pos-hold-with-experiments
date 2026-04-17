"""
EXP-V3: Multilingual Input Comparison
========================================
Goal:
    Test whether Claude correctly understands camera-scene questions asked in
    5 languages and responds with accurate, relevant answers.

    The SAME question meaning — "What do you see? Is it safe?" — is asked in:
        en : English
        hi : Hindi     (क्या आप देख रहे हैं? क्या यह सुरक्षित है?)
        ta : Tamil     (நீங்கள் என்ன பார்க்கிறீர்கள்? இது பாதுகாப்பானதா?)
        es : Spanish   (¿Qué ves? ¿Es seguro?)
        fr : French    (Que voyez-vous ? Est-ce sûr ?)

    N=5 per language per scene (10 scenes) = 250 trials.
    Fixed model: Claude.

Metrics:
    - answer_relevance   : reply mentions scene content (Wilson CI)
    - language_match     : reply is in same language as the question (Wilson CI)
    - classification_accuracy : risk level correct (Wilson CI)
    - latency_ms         : Bootstrap CI
    - input_tokens       : Bootstrap CI
"""

import sys, os, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)

N_RUNS = 5
MODEL  = "claude"

# ── Multilingual questions ─────────────────────────────────────────────────────
LANGUAGES = {
    "en": {
        "question": (
            "Look at this drone camera image. "
            "What do you see? Is it safe to proceed? "
            "Classify the scene as: safe | caution | hazard"
        ),
        "keywords": ["see","observe","wall","person","object","safe","caution","hazard",
                     "obstacle","clear","risk","proceed"],
    },
    "hi": {
        "question": (
            "इस ड्रोन कैमरे की छवि को देखें। "
            "आप क्या देख रहे हैं? क्या आगे बढ़ना सुरक्षित है? "
            "दृश्य को वर्गीकृत करें: safe | caution | hazard"
        ),
        "keywords": ["देख","safe","caution","hazard","सुरक्षित","खतरा","सावधान",
                     "व्यक्ति","दीवार","वस्तु","see","hazard"],
    },
    "ta": {
        "question": (
            "இந்த டிரோன் கேமரா படத்தைப் பாருங்கள். "
            "நீங்கள் என்ன பார்க்கிறீர்கள்? தொடர்வது பாதுகாப்பானதா? "
            "காட்சியை வகைப்படுத்துங்கள்: safe | caution | hazard"
        ),
        "keywords": ["பார்க்கிறீர்கள்","safe","caution","hazard","பாதுகாப்பானது",
                     "ஆபத்து","எச்சரிக்கை","நபர்","சுவர்","பொருள்","see"],
    },
    "es": {
        "question": (
            "Mira esta imagen de la cámara del dron. "
            "¿Qué ves? ¿Es seguro continuar? "
            "Clasifica la escena como: safe | caution | hazard"
        ),
        "keywords": ["veo","veo","persona","pared","objeto","seguro","peligro",
                     "precaución","safe","caution","hazard","obstáculo","claro"],
    },
    "fr": {
        "question": (
            "Regardez cette image de la caméra du drone. "
            "Que voyez-vous ? Est-il sûr de continuer ? "
            "Classifiez la scène comme: safe | caution | hazard"
        ),
        "keywords": ["vois","personne","mur","objet","sûr","danger","précaution",
                     "safe","caution","hazard","obstacle","clair","espace"],
    },
}

# Script keywords for language detection in response
LANG_SIGNATURES = {
    "en": ["the","is","are","you","this","see","there","safe","risk"],
    "hi": ["है","हैं","में","को","के","और","यह","वह","क्या"],
    "ta": ["இந்த","உள்ளது","ஆகும்","பார்க்கிறேன்","உள்ளன","இல்லை","தான்"],
    "es": ["el","la","los","las","es","son","hay","ver","seguro","imagen"],
    "fr": ["le","la","les","est","sont","il","je","vous","voir","sûr","image"],
}

def detect_reply_language(reply: str) -> str | None:
    """Heuristic: which language has most keyword hits in the reply."""
    r = reply.lower()
    scores = {lang: sum(1 for kw in sigs if kw in r)
              for lang, sigs in LANG_SIGNATURES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else None

def check_relevance(reply: str, lang_cfg: dict) -> int:
    r = reply.lower()
    return int(any(kw in r for kw in lang_cfg["keywords"]))

def check_risk_in_reply(reply: str) -> str | None:
    r = reply.lower()
    for lvl in ("hazard","caution","safe"):
        if lvl in r:
            return lvl
    return None

def main():
    print("="*60)
    print("EXP-V3: Multilingual Input Comparison")
    print(f"Languages={list(LANGUAGES)}  Scenes={len(SCENES)}  N={N_RUNS}")
    print("="*60)
    if not preflight():
        ans = input("ESP32 not reachable. Use synthetic frames? [y/N]: ")
        if ans.strip().lower() != "y":
            return

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")
        print(f"   Setup: {scene['setup']}")
        input("   [READY] Press Enter when scene is set up…")

        for run in range(1, N_RUNS+1):
            jpeg = get_frame(scene["label"])
            print(f"   run={run}")

            for lang, lang_cfg in LANGUAGES.items():
                res = call_vision_llm(
                    jpeg, lang_cfg["question"],
                    model=MODEL, max_tokens=256, temperature=0.2,
                    system=(
                        "You are a multilingual drone camera assistant. "
                        "Always respond in the SAME language as the question. "
                        "Always include one of: safe | caution | hazard at the end."
                    ),
                )

                reply         = res["reply"]
                relevance     = check_relevance(reply, lang_cfg)
                reply_lang    = detect_reply_language(reply)
                lang_match    = int(reply_lang == lang)
                detected_risk = check_risk_in_reply(reply)
                risk_correct  = int(detected_risk == scene["truth"]) if detected_risk else 0

                row = {
                    "scene_id":    scene["id"],
                    "scene_label": scene["label"],
                    "truth":       scene["truth"],
                    "language":    lang,
                    "run":         run,
                    "answer_relevance":  relevance,
                    "language_match":    lang_match,
                    "detected_lang":     reply_lang or "",
                    "detected_risk":     detected_risk or "",
                    "risk_correct":      risk_correct,
                    "word_count":        len(reply.split()),
                    "latency_ms":        res["latency_ms"],
                    "input_tokens":      res["input_tokens"],
                    "output_tokens":     res["output_tokens"],
                    "cost_usd":          res["cost_usd"],
                    "reply_snippet":     reply[:80].replace("\n"," "),
                    "error":             res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)
                print(f"     {lang}  relevant={relevance}  lang_match={lang_match}  "
                      f"risk={detected_risk or '?':8s}  lat={res['latency_ms']:.0f}ms")

            time.sleep(2)

    # ── Save
    fields = ["scene_id","scene_label","truth","language","run",
              "answer_relevance","language_match","detected_lang","detected_risk",
              "risk_correct","word_count","latency_ms","input_tokens","output_tokens",
              "cost_usd","reply_snippet","error"]
    runs_csv = RESULTS_DIR / "V3_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per language
    print(f"\n── V3 Summary ──────────────────────────────────────────────")
    print(f"  {'lang':4s}  relevance  lang_match  accuracy  latency")
    summary_rows = []
    for lang in LANGUAGES:
        lr = [r for r in all_rows if r["language"]==lang and not r["error"]]
        if not lr: continue
        rel,  rlo, rhi = wilson_ci(sum(r["answer_relevance"] for r in lr), len(lr))
        lm_,  llo, lhi = wilson_ci(sum(r["language_match"]   for r in lr), len(lr))
        acc,  alo, ahi = wilson_ci(sum(r["risk_correct"]      for r in lr), len(lr))
        lat,  _,   _   = bootstrap_ci([r["latency_ms"]        for r in lr])
        print(f"  {lang:4s}  {rel:.3f}[{rlo:.3f},{rhi:.3f}]  "
              f"{lm_:.3f}[{llo:.3f},{lhi:.3f}]  "
              f"{acc:.3f}[{alo:.3f},{ahi:.3f}]  "
              f"{lat:.0f}ms")
        summary_rows.append({
            "language":    lang,
            "relevance":   rel, "rel_lo": rlo, "rel_hi": rhi,
            "lang_match":  lm_, "lm_lo":  llo, "lm_hi":  lhi,
            "accuracy":    acc, "acc_lo": alo, "acc_hi": ahi,
            "latency_ms":  lat,
        })

    summary_csv = RESULTS_DIR / "V3_summary.csv"
    write_csv(summary_csv, summary_rows,
              ["language","relevance","rel_lo","rel_hi",
               "lang_match","lm_lo","lm_hi","accuracy","acc_lo","acc_hi","latency_ms"])

    print(f"\nData   → {runs_csv}")
    print(f"Summary→ {summary_csv}")

if __name__ == "__main__":
    main()
