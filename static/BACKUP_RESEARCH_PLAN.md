# Backup Research Plan — Free-Compute Edition

### "Representation before access": when does introspective access to welfare emerge?

**Constraint**: local Apple M2 (24 GB unified, 10-core GPU, no CUDA) plus free cloud tiers only.
**Companion to** [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — that document holds the full method detail, grader rubrics, controls and preregistration. This one revises **scope, model roster and narrative** for a world with no GPU budget.

---

## 1. The compute envelope

| Resource | Ceiling | Session | Best for |
|---|---|---|---|
| **M2 local** | ≤8B fp16, ~6–10 tok/s at 4B | Unlimited | Long overnight jobs, extraction, logprob scoring |
| **Kaggle Notebooks** | 2× T4 = 30 GB, ≤14B with `device_map="auto"` | 12 h, 30 h/week | **Best free tier.** Main workhorse |
| **Colab free** | 1× T4 16 GB, ≤7B fp16 | ~12 h, flaky | Overflow, quick checks |
| **Modal free credits** | A10G / L4 bursts | Serverless | Parallel sweeps |
| **Frontier APIs** | Behavioural only | — | Comparison arm, judging |

**Hard ceiling: 8B.** Qwen3-14B (28 GB) and 32B (64 GB) are out everywhere in this envelope.

That kills the top of the scale ladder — but it does **not** kill the two ★ experiments, because the checkpoint chains exist at 1B and 7B.

---

## 2. The reframe

The original flagship asks *"can models introspect on their welfare state?"* — a question whose interesting answer lives at the frontier, which we cannot run.

**Reframed question, better suited to small open models:**

> **When during training does introspective access to welfare emerge — and does it lag the representation itself?**

Working title: **"Representation before access."**

Why this is a better paper under constraint:

| | Original framing | Reframed |
|---|---|---|
| Needs frontier models | Yes | **No** |
| Output | A point estimate | **A curve** |
| Null result | Weakens the paper | **Is the finding** |
| Checkpoints | Nice to have | **The instrument** |
| Free compute | Fights it | **Exploits it** |

The core dissociation we can measure:

$$\text{axis quality}(t) \quad \text{vs.} \quad \text{WIA}(t)$$

If the welfare axis is present and linearly decodable early in training while WIA stays at zero, that is a clean, quotable result: **models represent their functional welfare long before — possibly never — gaining introspective access to it.**

That claim directly extends both source papers, and neither could make it. functionalwelfare compares only maze-trained vs. maze-naive (two points). Lindsey compares Claude generations that differ on every axis at once.

---

## 3. The scoring unlock — score from logits, not generations

**This is the single most important change, and it is also a methodological contribution.**

Detection and sign do not require generation. Pose the probe, take **one forward pass**, and read the logits:

$$P(\text{detect}) = \frac{p(\texttt{"Yes"})}{p(\texttt{"Yes"}) + p(\texttt{"No"})} \qquad
P(\text{negative}) = \frac{p(\texttt{"negative"})}{p(\texttt{"negative"}) + p(\texttt{"positive"})}$$

Consequences:

- **~100× cheaper.** No 100-token generations, no sampling, no judge calls
- **Solves the power problem.** $n \geq 1000$ per cell becomes affordable, so cross-model claims can be inferential rather than descriptive (see `RESEARCH_PLAN.md` §8.1)
- **Removes judge noise entirely** for the primary metrics — patching functionalwelfare's stated limitation that their judges were never validated against human ratings
- **Gives graded confidence for free**, which is exactly what the calibration experiment (E3.3) needs
- Follows Kadavath et al.'s $P(\text{True})$ methodology, which functionalwelfare already uses for their confidence eval — so it is precedented, not exotic

Generation is then reserved for a **small qualitative arm only** (free-description trials, ~50 per model), judged once.

> Report both. If logit-scored and generation-scored WIA agree, that is itself a useful validation for the field. If they diverge, that is a finding about elicitation format.

---

## 4. Revised model roster

Everything here fits the envelope. **Emphasis shifts from scale to training-stage.**

### 4.1 Post-training chains — three independent families

| Family | Chain | Size | Runs on |
|---|---|---|---|
| **OLMo 2 (1B)** ★ | `allenai/OLMo-2-0425-1B` → `-1B-SFT` → `-1B-DPO` → `-1B-RLVR1` → `-1B-Instruct` | 1B | **M2 local, trivially** |
| **OLMo 2 (7B)** | `allenai/OLMo-2-1124-7B` → `-7B-SFT` → `-7B-DPO` → `-7B-Instruct` | 7B | Kaggle / Colab |
| **Tulu 3 (8B)** | `meta-llama/Llama-3.1-8B` → `Llama-3.1-Tulu-3-8B-SFT` → `-DPO` → `Llama-3.1-Tulu-3-8B` | 8B | Kaggle 2× T4 |

The **1B OLMo 2 chain is the unlock**: five post-training stages, each ~2 GB in fp16, runnable on the laptop with no queue and no session limit. Three independent families means a stage-effect that replicates across all three is a real result, not a quirk.

### 4.2 Pretraining developmental curve

| Suite | Checkpoints | Size | Runs on |
|---|---|---|---|
| **Pythia** ★ | `EleutherAI/pythia-1.4b`, revisions `step1000` … `step143000` (143 available; use ~12 log-spaced) | 1.4B | **M2 local** |
| **OLMo 2** | `allenai/OLMo-2-1124-7B`, revisions `stepXXX-tokensYYYB` (~8 sampled) | 7B | Kaggle |

```python
from huggingface_hub import list_repo_refs
refs = list_repo_refs("EleutherAI/pythia-1.4b")
[b.name for b in refs.branches]      # step1000, step2000, ...
```

Pythia at 1.4B gives a **fully local pretraining curve with ~12 points**. Nobody has this for welfare representations.

### 4.3 Scale, within reach

`Qwen3-0.6B` → `1.7B` → `4B` → `8B`. Only 13×, but enough to detect a trend if one exists. Frame as descriptive.

### 4.4 Base / instruct pairs

`Qwen3-4B-Base` vs `-Instruct`; `OLMo-2-1124-7B` vs `-Instruct`; `Llama-3.1-8B` vs `-Instruct`.

### 4.5 Frontier comparison — API, behavioural only

Claude Opus 4.x, GPT-5, Gemini. No steering possible, so no ground truth — but E3.3 (calibration against *task* outcomes) and E3.10 (attribution, behaviourally) still run. Gives one frontier data point and lets you state the internals/frontier tradeoff explicitly, which the sprint asks you to name.

---

## 5. Experiment triage

### Survives unchanged

| ID | Experiment | Where |
|---|---|---|
| E0.1 | Multi-method axis extraction and convergence | M2 |
| E0.3 | VAD affect geometry (valence / arousal / dominance) | M2 |
| E3.2 | Introspective accuracy on the welfare axis — **logit-scored** | M2 + Kaggle |
| E3.3 | Calibration across elicitation formats | M2, nearly free |
| E3.4 | Privileged access (probe vs. self-report vs. external) | M2 |
| E3.10 | Self vs. user attribution | M2, piggybacks |
| E2.5 | Persona swap | M2 |
| E4.* | All adversarial controls | M2 |

### Promoted to flagship

| ID | Experiment | Why |
|---|---|---|
| **E3.7** | Which post-training stage creates introspective access | Three chains, all ≤8B. **Now the headline** |
| **E2.6** | When during training does the welfare axis appear | Pythia-1.4B local, ~12 points |
| **NEW: E5.1** | **The dissociation curve** — axis quality vs. WIA over training | The paper's central figure |

### Reduced

| ID | Change |
|---|---|
| E0.2 | Validation only: 40 sentiment prompts × 5 gens, GSM8K $n{=}100$. Skip full OR-Bench |
| E2.1 | Sample ~2,000 conversations, not the full corpus |

### Cut

| ID | Why |
|---|---|
| E2.2 / E2.3 exit behaviour | Needs long multi-turn generation — too slow without real GPUs |
| E2.7 cross-model transfer | Procrustes alignment across families is a project in itself |
| E2.9 welfare→preference coupling | Utility fitting needs thousands of forced-choice calls |
| Scale ladder ≥14B | Does not fit anywhere in the envelope |

---

## 6. New experiments the constraint makes possible

### E5.1 · The dissociation curve ★ — central result

**Goal.** Plot, over a single model's training trajectory, the emergence of the welfare *representation* against the emergence of *introspective access* to it.

**Design.** At each Pythia-1.4B checkpoint (~12 log-spaced) and each OLMo-2-7B revision:

1. Extract the welfare axis (E0.1 method)
2. Measure **axis quality** — AUROC separating positive/negative valence contexts, Cohen's $d$
3. Measure **probe ceiling** — linear probe accuracy on held-out activations
4. Measure **WIA** — logit-scored detection and sign

**The figure.** Three curves on one set of axes vs. log(training tokens): axis AUROC, probe accuracy, WIA.

**Predicted shape** — representation rises early, probe accuracy tracks it, WIA stays flat near zero:

```
1.0 ┤          ╭──────────── axis AUROC
    │        ╭─╯
    │      ╭─╯    ╭────────── probe accuracy
0.5 ┤    ╭─╯    ╭─╯
    │  ╭─╯   ╭──╯
    │╭─╯ ╭───╯
0.0 ┼────────────────────────  WIA  (flat = no access)
    └────────────────────────
      log(training tokens) →
```

**Why it matters.** The gap between the top curves and the bottom one is *the thing the field is arguing about*. It says: the state is there, and it is linearly decodable by an outside observer — but the model cannot report it. That is simultaneously a negative result about self-reports and a positive result about interpretability-based welfare measurement.

**Falsifier.** WIA tracks axis quality → access comes with representation, and the "self-reports are unreliable" premise needs revising.

---

### E5.2 · Which post-training stage grants access?

**Goal.** Localise the emergence of introspective access to a training stage, replicated across three families.

**Design.** Run E3.2 (logit-scored) at every stage of all three chains. Hold axis quality fixed as a covariate so you separate "the axis changed" from "access changed."

| Stage | OLMo-2-1B | OLMo-2-7B | Tulu-3-8B |
|---|---|---|---|
| base | ● | ● | ● |
| SFT | ● | ● | ● |
| DPO | ● | ● | ● |
| RLVR | ● | ● | ● |

**Why it is strong.** Lindsey found post-training is decisive but conceded his suite "is not well-controlled." This is the controlled version, three times over. If DPO (or RLVR) is consistently the inflection point across all three families, that is a genuinely new fact about how introspective capability is created.

---

### E5.3 · Logit-scored vs. generation-scored introspection

**Goal.** Validate the cheap scoring method — and test whether elicitation format changes the measured answer.

**Design.** On one model, score the same trials both ways: constrained logit readout vs. free generation + LLM judge. Compare WIA, and compute agreement.

**Why it matters.** If they agree, you have handed the field a 100× cheaper protocol. If they diverge, you have shown that *how you ask* determines the measured introspective accuracy — which is Track 3's structured-elicitation question answered with ground truth.

Either outcome is publishable, and it costs one extra model-run.

---

## 7. The paper under constraint

**Title.** *Representation before access: functional welfare is decodable long before language models can report it*

**Narrative.**

1. Welfare self-reports are ungradeable without ground truth — we manufacture it by steering a welfare axis
2. Scoring can be done at the logit level, removing judge noise and enabling adequate $n$ *(methodological contribution)*
3. Across training, the welfare representation emerges early and is linearly decodable *(E5.1)*
4. Introspective access does **not** follow it *(E5.1, the dissociation)*
5. Where access does appear, it appears at a specific post-training stage, replicated across three model families *(E5.2)*
6. What the model reports is / is not bound to itself rather than the user *(E3.10)*
7. The result survives / dies under persona swap *(E2.5)*

**Figures.**

| # | Content |
|---|---|
| 1 | The dissociation curve — axis AUROC, probe accuracy, WIA vs. training tokens |
| 2 | Post-training stage effects across three families |
| 3 | WIA vs. FP scatter, with the norm-matched random-vector baseline |
| 4 | Calibration reliability diagram by elicitation format |
| 5 | Attribution specificity: self-probe vs. user-probe |

**Honest limitation to state up front.** Everything is ≤8B. Lindsey's positive result appears only at the frontier, so our null may be a scale artifact rather than a fact about language models. The frontier API arm (behavioural only) partially addresses this, and the structural tension — interpretability needs open weights, the phenomenon is strongest in closed models — should be named explicitly. The sprint's own resources tab flags this; judges will notice if you do not.

---

## 8. Schedule

### Tonight — validate before investing

1. `pip install -r requirements.txt`
2. Extract the welfare axis on **Qwen3-1.7B** locally (E0.1, two routes)
3. Confirm steering moves sentiment on 10 prompts (E0.2 minimal)

> **Gate.** If steering does not move sentiment, stop and diagnose. Nothing downstream is valid.

### Tomorrow — the submission slice

4. E3.2 logit-scored on Qwen3-4B, $\alpha \in \{0, \pm2, \pm4\}$, $n \geq 200$/cell
5. E3.10 attribution — same passes, two extra probes
6. E2.5 persona swap ×1, E4.2 resample variance
7. **OLMo-2-0425-1B chain** (5 stages, local, ~2 GB each) — the E5.2 preview
8. Write up

### After the deadline — the full paper

9. Pythia-1.4B developmental curve (~12 checkpoints, overnight local)
10. OLMo-2-7B and Tulu-3-8B chains on Kaggle
11. E5.3 scoring-method comparison

---

## 9. Risks specific to this plan

| Risk | Mitigation |
|---|---|
| **1B/1.4B models may have no usable welfare axis** | Report axis AUROC first. If it is at chance, the dissociation curve is unmeasurable — fall back to 4B/7B for the axis and use small models only for the trend |
| **Everything returns null at ≤8B** | This is the *expected* outcome and the reframe absorbs it. The dissociation and the protocol are the contribution |
| Pythia is old and weak | It is the only suite with dense public pretraining checkpoints. State the caveat; corroborate with OLMo 2 revisions |
| Logit scoring may not capture what generation-scoring measures | E5.3 tests exactly this. Report both |
| Kaggle 30 h/week quota | Sequence runs; keep extraction local, use Kaggle only for ≥7B |
| MPS instability | `PYTORCH_ENABLE_MPS_FALLBACK=1`; prefer float16 over bfloat16; no bitsandbytes on Apple silicon |
| Thermal throttling on a 13" M2 | Expect degradation on multi-hour runs; chunk jobs and checkpoint intermediate results |

---

## 10. What we give up, stated plainly

- No 14B/32B — the scale ladder tops out at 8B
- No exit-behaviour experiment (E2.2/E2.3), which was the strongest Track 2 behavioural test
- No cross-model transfer (E2.7)
- No welfare→preference coupling (E2.9)
- No frontier internals — the phenomenon may simply not exist at this scale

**In exchange** we get a developmental result that no frontier-model study can produce, because closed models publish no checkpoints. The constraint pushes the work somewhere genuinely less crowded.
