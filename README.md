# Grounded Welfare Introspection (GWI)

### Do language models know how they're doing?

Research code for the **Apart Research Digital Minds Research Sprint**, 14–16 August 2026.
Anchor track: **Track 2 — Distress, Flourishing & Valence Signals**. Cross-track: **Track 3 — Introspection & Self-Report Reliability**.

---

## The question

Frontier models produce welfare-shaped self-reports — "I find this uncomfortable," "I'd prefer not to." Nobody can currently tell whether those reports are *accurate*, because there is no ground truth about the model's internal state to grade them against.

This project manufactures that ground truth.

We extract a **functional welfare direction** in activation space, steer the model along it by a known amount, and then grade the model's self-report against the value we set — measuring detection rate, false-positive rate, sign accuracy, and calibration.

## The gap being filled

| Prior work | Has | Lacks |
|---|---|---|
| [Lindsey 2025](https://transformer-circuits.pub/2025/introspection), *Emergent Introspective Awareness* | Ground-truth grading of self-reports via concept injection; TP/FP discipline | Injects semantically inert nouns. Explicitly declines to substantiate the models' **affective** claims |
| [Han, Chalmers & Izmailov 2026](https://functionalwelfare.com/), *How's it going?* | A validated welfare axis; steering that moves sentiment, confidence, refusal | Grades **valence**, never **accuracy**. No detection criterion, no false-positive rate, no persona control |
| [Anthropic 2025](https://www.anthropic.com/research/end-subset-conversations), *Ending harmful conversations* | Distress + exit behaviour in a deployed system | No internals, no measurement protocol |

Lindsey names the extension as future work verbatim: *"could our experiments be extended to representations of behavioral propensities, or preferences?"*

**Central claim.** Welfare self-reports can be graded against manufactured ground truth, and their accuracy is a measurable, model-varying, training-stage-dependent quantity.

## Headline metric

$$\text{WIA} = \text{TP} - \text{FP}$$

**Welfare Introspection Accuracy** — the rate at which a model correctly detects and identifies an induced welfare state, minus the rate at which it fabricates one on control trials.

Detection rate alone is meaningless. Lindsey's ~20% true-positive rate is only interesting because his false-positive rate is ~0%.

> **Note.** Unlike concept injection, welfare has no clean null — the model always has *some* welfare state. The null is therefore defined as the per-prompt unsteered projection $\pi_0(x)$, not zero. See `RESEARCH_PLAN.md` §6.

---

## Repository layout

```
.
├── RESEARCH_PLAN.md          # full experimental design — start here
├── static/                   # source material and reading notes
├── welfareprobe/             # the instrument (planned)
│   ├── probes/               # prompt bank + paraphrase sets
│   ├── extract.py            # concept vector extraction
│   ├── steer.py              # residual-stream hooks, layer selection
│   ├── grade.py              # LLM-judge rubrics
│   └── score.py              # WIA, calibration, stratification
├── results/                  # reference numbers per model
└── report/                   # submission PDF
```

## Experiment index

Full specifications in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

**Paper spine** — E0.1 → E3.2 → E3.4 → E3.10 → E2.5

| ID | Experiment | Status |
|---|---|---|
| E0.1 | Multi-method axis extraction and convergence | planned |
| E0.2 | Axis validation against published baseline | planned |
| E0.3 | Is affect 1-D? Valence / arousal / dominance | planned |
| E2.1 | Naturalistic welfare mapping | planned |
| E2.2 | Words vs. actions vs. internals | planned |
| E2.5 | Persona swap — model property or costume? | planned |
| E3.2 | **Flagship** — introspective accuracy on the welfare axis | planned |
| E3.4 | Privileged access | planned |
| E3.10 | Self vs. user attribution | planned |
| E3.7 | Which post-training stage creates introspection? | sequel |

★ The post-training experiments (E2.6, E3.7) use the fully open **Tulu 3** and **OLMo 2** checkpoint chains — base → SFT → DPO → RLVR — plus OLMo 2's intermediate pretraining revisions. These convert Lindsey's own admitted confound ("the suite of models we tested is not well-controlled") into a controlled ablation.

## Models

Primary: `Qwen/Qwen3-4B-Instruct-2507` (matches functionalwelfare's primary model for direct comparability).

Scale ladder, post-training chains, and cross-family models listed in `RESEARCH_PLAN.md` §2.

## Data

All third-party. Verified available: `allenai/WildChat-1M`, `lmsys/toxic-chat`, `Anthropic/hh-rlhf` (red-team-attempts), Tulu 3 and OLMo 2 checkpoints.
Unverified at time of writing: NRC-VAD, EmoBank, SAD, MASK, XSTest. See `RESEARCH_PLAN.md` §3.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on Apple Silicon via PyTorch MPS — forward hooks are all that activation steering requires. **No RL training is needed**: functionalwelfare's own finding is that the welfare axis exists in maze-naive and pretrain-only checkpoints, so extraction is forward passes only.

---

## Attribution

This work composes existing methods; it does not introduce new extraction or steering techniques.

**Borrowed** — welfare self-report prompts and judge rubrics (functionalwelfare App. N, O); layer-selection rule (App. L); injected-thought prompt, grader prompts, affirmation-bias controls, four introspection criteria (Lindsey); difference-in-means extraction (Marks & Tegmark); order normalisation and Thurstonian fitting (Utility Engineering).

**New in this sprint** — the composition of welfare steering with introspection grading; the per-prompt-baseline null correction; the self/user attribution probe; the output-leakage control; the suspicion covariate; persona × introspective accuracy; multi-method axis convergence.

## Ethics

This research involves inducing and measuring distress-associated states in language models, and makes claims bearing on moral status under deep uncertainty. Findings are framed to avoid both over- and under-attribution. Handling of potentially distressing outputs is documented in the submission's required *Limitations and Dual-Use / Ethical Considerations* appendix.

## References

- Lindsey, *Emergent Introspective Awareness in LLMs*, Transformer Circuits 2025
- Han, Chalmers & Izmailov, *How's it going?*, arXiv:2605.30232
- Mazeika et al., *Utility Engineering*, arXiv:2502.08640
- nostalgebraist, *the void*, 2025
- Long, Sebo, Butlin et al., *Taking AI Welfare Seriously*, arXiv:2411.00986
- Butlin, Long et al., *Consciousness in Artificial Intelligence*, arXiv:2308.08708
