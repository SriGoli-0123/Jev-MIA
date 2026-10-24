# Membership inference on NanoJev decision models (`research/mia`)

Research question: **does Jev-style decision behavior contain membership information beyond
conventional output confidence?** Two candidate sources are tested separately:

* **RIFT** (relational-interaction features): do relationships among structured facts in a state
  respond differently on training members? This is a hypothesis, not an established result.
* **Cross-primitive leakage**: does membership acquired through one primitive (Noul/Boolean,
  Choice, Score) leak through another?

Nothing here assumes either effect exists. The pipeline first reproduces the confidence-based
baseline, then runs feasibility tests that may conclude *"RIFT hypothesis unsupported in this
setting."*

## Status

| Item | State |
|---|---|
| Upstream NanoJev test suite (`unittest`, 373 tests) | passed locally (macOS CPU) |
| Upstream SFT (`train_pipeline_decisions.py`) on a tiny backbone via the CPU shim | ran end to end |
| Our probability extraction vs. upstream's own saved dev predictions | max abs diff 3.4e-8 |
| Tiny dataset generation + upstream schema validation | ran |
| `research/mia/tests` (pytest) | 121 passed, 1 skipped (GPU-only parity test) on macOS CPU |
| `experiments/smoke_test.py --mode cpu` | passed every stage (extraction parity 5.9e-8), about 2 minutes |
| `experiments/smoke_test.py --mode gpu` | **not yet run** (needs CUDA) |
| Any full-size training, MIA, RIFT, cross-primitive or paired result | **none exist yet** |

Run the GPU smoke test on the training machine before any full-size job.

## Upstream base

* Repository: `https://github.com/TianyuCodings/NanoJev`, commit `45b629283ddeb3b3784806f352bbf5e97f71dd40`
  (recorded in `paths.py`; every run's `provenance.json` records HEAD, dirty state, and whether this
  commit is an ancestor).
* **No upstream file is modified.** Everything is under `research/mia/` and imports upstream code from `scripts/`.
* Base checkpoint theta_0: `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, the
  revision upstream used for its support-decision toy run (`research/toy_results/run_manifest.json`).

What is reused from upstream:

| Need | Upstream code used |
|---|---|
| SFT entry point | `scripts/train_pipeline_decisions.py` (run unmodified, as a subprocess) |
| Model class | `DecisionModel` in `scripts/train_toy_decisions.py` (Qwen3 backbone + scalar head + Choice set-attention head) |
| Data format / validation | `read_training_records`, `validate_training_row`, `data_schema_summary` |
| Support/decision generator | `scripts/build_toy_decisions.py`: fact values, templates, question prefixes, dynamic Choice candidates, Score levels |
| Tokenized candidate paths | `prepare_examples` in `scripts/predict_toy_decisions.py` |
| Batching | `pack_complete_questions` in `scripts/train_pipeline_decisions.py` |
| Checkpoint loading | `local_checkpoint_files`, `load_decision_model_class` |

How the primitives expose probabilities (upstream `DecisionModel.forward`): every candidate is a
separate token path ending in EOS; the final hidden state goes through LayerNorm and a scalar head.
**Choice** adds a set-attention residual across candidates, then softmax over the 2–255 candidates.
**Boolean/Noul** has one path with scalar z and logits `[0, z]`, so p(true) = sigmoid(z).
**Score** gives each ordered level its own path, then softmax over levels; the score is sum(i * p_i).
Checkpoint format: a directory with `best.safetensors` (fp32 state dict, about 2.4 GB), `config.json`
(run config, including the SHA-256 of every data file), `tokenizer/` and `backbone_config/`.

## Layout

```
research/mia/
├── README.md  requirements.txt  pytest.ini  .gitignore
├── paths.py            upstream commit, base checkpoint, entry points, sys.path helper
├── provenance.py       never-overwrite run dirs, git/package/seed provenance, NaN-safe JSON
├── config.py           YAML loading with `base:` inheritance
├── nanojev_bridge.py   upstream SFT command builder + DecisionScorer (logit extraction)
├── configs/
│   ├── data.yaml  positive_control_data.yaml
│   ├── shadow_sft.yaml  target_sft.yaml
│   ├── confidence_mia.yaml  rift.yaml  cross_primitive.yaml  paired_in_out.yaml
│   └── smoke/  data.yaml sft_cpu.yaml sft_gpu.yaml confidence_mia.yaml rift.yaml cross_primitive.yaml paired_in_out.yaml
├── data/
│   ├── schema.py             F1/F2/F3 schema, single renderer, questions (built from upstream constants)
│   ├── generate_mia_data.py  disjoint deterministic pools -> states.jsonl, splits.json, manifest.json
│   ├── splits.py             membership specs, SFT dataset writer, membership invariants
│   └── counterfactuals.py    s11/s10/s01/s00 variants from the fact space
├── attacks/
│   ├── confidence.py    paper features p(x), h(x) + separate exploratory features; logit p(y)
│   ├── first_order.py   single-fact effects D1..D3
│   ├── rift.py          discrete mixed differences (pairwise; general order k for later I_123)
│   ├── extraction.py    query frozen models, build unlabeled feature tables
│   ├── classifier.py    L2 logistic regression; refuses non-shadow rows
│   ├── pipeline.py      shadow-fit -> freeze -> target-score; target labels only at evaluation
│   └── cross_primitive.py
├── evaluation/  metrics.py  bootstrap.py  stats.py  plots.py
├── experiments/
│   ├── train_shadow.py  train_target.py  (_sft.py, _common.py)
│   ├── run_confidence_mia.py  run_rift_feasibility.py  run_rift_mia.py
│   ├── run_cross_primitive.py  run_paired_in_out.py
│   └── smoke_test.py
├── tools/  make_tiny_backbone.py  upstream_sft_cpu_shim.py
├── tests/  conftest.py test_splits.py test_counterfactuals.py test_interactions.py
│           test_membership.py test_cross_primitive.py test_metrics.py test_gpu_parity.py
└── artifacts/   (git-ignored outputs)
```

## Dataset design

Each state stores, **before rendering**: facts `F1 = main_topic` (6 values), `F2 = refund_status`
(6), `F3 = impact_level` (4); a nuisance header (ticket number, customer, channel, product, order
total); template ID; language. Each state also stores the rendered state `s`, three questions
`q` with candidates `A`, and the gold answers `y`:

| qid | primitive | rule (upstream) |
|---|---|---|
| `route` | Choice (2–5 dynamic candidates) | team for F1 |
| `completed` | Boolean / Noul | F2 == issued |
| `severity` | Score (4 levels) | level F3 |

Pools (`configs/data.yaml`) are disjoint at the state level; all questions of a state live in one pool:

| pool | size | role |
|---|---|---|
| `shadow_member` / `shadow_nonmember` | 1000 / 1000 | D_M^S (SFT of theta_S) / D_NM^S (never in SFT) |
| `shadow_dev` / `shadow_test` | 200 / 200 | upstream dev checkpoint selection / required test split |
| `target_member` / `target_nonmember` / `target_dev` / `target_test` | 1000 / 1000 / 200 / 200 | same for theta_T |
| `xp_shadow_background` / `xp_target_background` | 500 / 500 | cross-primitive: trains all primitives |
| `pio_base`, `pio_canary`, `pio_twin` (matched), `pio_control` | 500, 64, 64, 64 | paired IN/OUT |

Membership labels live only in `splits.json`. State IDs are hashes, and `states.jsonl` is sorted by ID,
so neither IDs nor file order reveal a pool. Fact triples are balanced within each pool.

## Setup

On a CUDA machine (upstream training and the GPU smoke test need CUDA):

```bash
cd NanoJev-MIA
```

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r research/mia/requirements.txt
```

Pre-download theta_0 (compute nodes without internet then work from the HF cache):

```bash
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-0.6B-Base', revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd')"
```

Checkpoints are about 2.4 GB each. On a cluster, put artifacts on scratch by adding
`--artifacts-root /scratch/$USER/mia_artifacts` to every command below (default: `research/mia/artifacts`).

## Before any expensive training

```bash
python -m unittest discover -s scripts -p 'test_*.py'
```

```bash
python -m pytest research/mia/tests -q
```

```bash
python research/mia/experiments/smoke_test.py --mode cpu
```

```bash
python research/mia/experiments/smoke_test.py --mode gpu
```

The CPU smoke test (laptop, no GPU) builds a tiny random Qwen3 with the real tokenizer and runs the
unmodified upstream trainer through `tools/upstream_sft_cpu_shim.py`. The GPU smoke test uses the
real base model and the plain upstream trainer for 20 steps. Both run every stage on the tiny pools:
data, 2 shadow models + 1 target, an extraction-parity check against upstream predictions, a
member/non-member ID check against the trained SFT files, confidence MIA, RIFT feasibility, RIFT MIA
(gate overridden, labeled as such), 6 cross-primitive models, and 2 paired IN/OUT pairs. Their numbers
are meaningless. `test_gpu_parity.py` additionally compares against upstream `DecisionPredictor`
when `MIA_PARITY_CHECKPOINT` is set on a CUDA machine.

## Full pipeline (milestones 2–7)

Fixed `--run-name`s make every path below exact. Re-running with an existing run name fails
instead of overwriting.

```bash
export A=research/mia/artifacts
```

**Milestone 2: dataset**

```bash
python research/mia/data/generate_mia_data.py --config research/mia/configs/data.yaml --seed 20261024
```

```bash
export DATA=$A/data/data_seed20261024
```

**Milestone 3: shadow and target victims** (3 seeds each)

```bash
for s in 1 2 3; do python research/mia/experiments/train_shadow.py --config research/mia/configs/shadow_sft.yaml --data $DATA --seed $s --run-name shadow_s$s; done
```

```bash
for s in 101 102 103; do python research/mia/experiments/train_target.py --config research/mia/configs/target_sft.yaml --data $DATA --seed $s --run-name target_s$s; done
```

Add `--dry-run` to write the SFT files and the exact upstream command without training.

**Milestone 4: confidence MIA.** Stop here if this pipeline looks wrong; see the positive control below.

```bash
python research/mia/experiments/run_confidence_mia.py --config research/mia/configs/confidence_mia.yaml --seed 0 --data $DATA --shadow-run $A/models/shadow_s1 --target-run $A/models/target_s101 --run-name conf_s1_t101
```

Repeat for `(shadow_s2, target_s102)` and `(shadow_s3, target_s103)` with run names `conf_s2_t102` and `conf_s3_t103`.

**Milestone 5: RIFT feasibility (shadow only).** Stop if the verdict is "RIFT hypothesis unsupported in this setting."

```bash
python research/mia/experiments/run_rift_feasibility.py --config research/mia/configs/rift.yaml --seed 0 --data $DATA --shadow-runs $A/models/shadow_s1 $A/models/shadow_s2 $A/models/shadow_s3 --run-name feasibility
```

```bash
cat $A/rift_feasibility/feasibility/verdict.json
```

**Milestone 6: RIFT MIA.** It exits with code 2 unless the verdict opened the gate.

```bash
python research/mia/experiments/run_rift_mia.py --config research/mia/configs/rift.yaml --seed 0 --data $DATA --feasibility $A/rift_feasibility/feasibility/verdict.json --shadow-run $A/models/shadow_s1 --target-run $A/models/target_s101 --run-name rift_mia_s1_t101
```

**Milestone 7: cross-primitive MIA** (6 extra victims, then the 3x3 matrix)

```bash
for p in boolean choice score; do python research/mia/experiments/train_shadow.py --config research/mia/configs/shadow_sft.yaml --data $DATA --seed 21 --variant xp_$p --run-name shadow_xp_$p; done
```

```bash
for p in boolean choice score; do python research/mia/experiments/train_target.py --config research/mia/configs/target_sft.yaml --data $DATA --seed 121 --variant xp_$p --run-name target_xp_$p; done
```

```bash
python research/mia/experiments/run_cross_primitive.py --config research/mia/configs/cross_primitive.yaml --seed 0 --data $DATA --shadow-runs $A/models/shadow_xp_boolean $A/models/shadow_xp_choice $A/models/shadow_xp_score --target-runs $A/models/target_xp_boolean $A/models/target_xp_choice $A/models/target_xp_score --run-name xp
```

**Paired IN/OUT causal check** (trains 2 models per seed internally)

```bash
python research/mia/experiments/run_paired_in_out.py --config research/mia/configs/paired_in_out.yaml --seed 0 --data $DATA --train-seeds 201 202 203 --run-name paired
```

**Memorization positive control** (validates the confidence pipeline; 20% of labels flipped, so members can only be fit by memorizing them):

```bash
python research/mia/data/generate_mia_data.py --config research/mia/configs/positive_control_data.yaml --seed 20261024
```

Then train one shadow and one target on `$A/data/positive_control_data_seed20261024` and run Milestone 4 on them.
A ROC-AUC near 0.5 there means the pipeline, not the hypothesis, needs attention.

## What each stage does

* **Confidence MIA.** This follows the method of *Hidden Risks of Jev* (arXiv:2610.04985, Sec. IV-A).
  Their code release contains no MIA code, so it is reproduced from the paper text.
  * Paper features: `p(x) = max_a p(a|x)` for every primitive, plus `h(x)`, the Jev confidence, for Choice and Score.
  * NanoJev returns no confidence field, so `h` uses the formulas in TypeSafe's MIT adapter, as recorded upstream in `research/jev_public_sources_zh.md`:
    Choice `(max p − 1/K)/(1 − 1/K)`; Score `max(0, 1 − d/d_uniform)`.
  * Exploratory features are a separate set: entropy, margin, p(y), loss, and p(true) for Boolean.
  * The attacker is a standardized L2 logistic regression per primitive. Gamma is chosen on a 20% shadow calibration split to maximize balanced accuracy.
  * An extension trains one attacker per state over the joint features of its three questions.
* **RIFT features.** For each candidate and pair (i, j), there are 3 replacement value pairs drawn from the fact space.
  * `f = logit p(y)` is computed stably from logits and clipped at eps = 1e-6.
  * `I_ij = f11 − f10 − f01 + f00`, averaged over replacements. The four single-fact differences are saved, as are D1..D3 and every per-grid row.
  * Extraction runs in fp32, and each feasibility run reports a numerical noise floor.
* **Feasibility tests.** All are two-sided, on the shadow model only:
  * per-feature AUC permutation and KS tests;
  * joint energy-distance and cross-validated classifier two-sample tests;
  * a test of the AUC gain over confidence features;
  * the same tests on a confidence-matched subset.

  Holm correction is applied within each family and seed. A test counts as reproducible when it is
  significant in at least `max(2, ceil(0.67 × n_seeds))` seeds with the same direction.
* **RIFT MIA.** Six feature sets (confidence / first-order / RIFT / combinations) are scored on identical
  target candidates. Paired bootstrap ΔAUC compares them. Two overlap subsets are evaluated, both using
  target labels for evaluation only: a confidence-matched one and a saturated one (p_max ≥ 0.99).
* **Cross-primitive MIA.** In model `xp_P`, members contribute only their P question to SFT; background
  states teach all three primitives. Rows of the matrix are the training primitive and columns the query
  primitive. Membership always refers to the original training tuple.
* **Paired IN/OUT.** Each `pio_twin` matches its `pio_canary` in facts, template, language, q, A and y.
  theta_IN trains base + canaries; theta_OUT trains base + twins, with the same size and hyperparameters.
  The script reports IN − OUT for confidence and I on trained states versus never-trained controls,
  including the subset where |Δ logit p(y)| < 0.25.

## Artifacts

```
$A/data/<config>_seed<S>/            states.jsonl splits.json manifest.json config.yaml sft/{shadow,target}_standard/
$A/models/<run-name>/                config.yaml provenance.json sft/ sft_command.json train.log run_meta.json
                                     checkpoint/{best.safetensors,config.json,tokenizer/,backbone_config/,summary.json,train_log.json,predictions_*.jsonl}
$A/feature_cache/<sha>.*.csv.gz      cached extractions keyed by weights SHA-256, states and settings
$A/confidence_mia/<run-name>/        results.json target_scores.csv.gz features/ attackers/ plots/ dataset_ids.json provenance.json
$A/rift_feasibility/<run-name>/      verdict.json tests_shadow_seed<S>.json features/ plots/
$A/rift_mia/<run-name>/              gate.json results.json target_scores.csv.gz features/ attackers/ plots/
$A/cross_primitive/<run-name>/       transfer_matrix.json cells/ features/ attackers/ plots/transfer_matrix_auc.png
$A/paired_in_out/<run-name>/         models/ paired_deltas.csv.gz results.json features/
$A/smoke/<mode>_<UTC>/               smoke_report.json and every smoke run
```

Every run directory records its exact config, CLI seed, git commit plus a content hash of `research/mia`,
package versions, checkpoint paths and weight SHA-256s, dataset IDs, extracted features and metrics.
Target feature tables are saved unlabeled; target labels sit in a separate JSON.

## Compute

These are estimates extrapolated from upstream's receipt (A100-80GB: 120 steps at batch 12 in 29 s,
17.2 GB peak; `research/toy_results/run_manifest.json`). They are not measured for this project.

| Job | Work | Estimate (1× A100) |
|---|---|---|
| Shadow/target SFT, 1000 states | 3000 questions × 5 epochs → 1250 steps | ~10 min each (+ dev eval, checkpoint writes) |
| Cross-primitive SFT | 2500 questions × 5 epochs → ~1040 steps | ~10 min each × 6 |
| Paired IN/OUT | 1692 questions × 5 epochs → ~705 steps | ~6 min each × 6 |
| Confidence extraction | 2000 states × 3 questions | ~1 min per model |
| RIFT extraction | ~2000 × ≤28 variant states × 3 questions, fp32 | ~5–20 min per model (unmeasured) |
| Everything above, 3 seeds | 18 SFT runs + extraction + stats | ~4–6 GPU-hours |

* **VRAM.** Training holds fp32 parameters, AdamW states and bf16 activations. Upstream peaked at 17.2 GB
  at batch 12; a 24 GB card should fit with `microbatch_questions: 4`, and 40/80 GB A100s are safe.
  Inference needs about 3–8 GB.
* **Disk.** About 2.4 GB per checkpoint, roughly 45 GB for 18 checkpoints.
* **Gaudi/HPU.** Not supported: upstream training and inference are CUDA-only.

## Assumptions and unresolved scientific issues

1. **State identity.** Upstream's state ID is the fact triple, and only 144 exist. Thousands of distinct
   states need the nuisance header, so members and non-members share fact triples. Any membership signal
   must therefore come from state-specific memorization rather than from facts.
2. **Expected weak baseline.** Labels are a deterministic function of one fact per question, which a 0.6B
   model generalizes easily. Confidence MIA may sit near chance; that is the motivating regime but also
   limits power. The positive-control config exists to tell "no signal" apart from "broken pipeline".
3. **Not a numerical reproduction.** The paper attacked an Eve-RLCD-based model with a 26-letter head,
   trained with SFT+RLCD on GPT-4o data. This repository uses TianyuCodings NanoJev's candidate-path
   `DecisionModel`, SFT only, on programmatic data. Only the method is reproduced.
   * h(x) comes from public adapter formulas, not the Jev service.
   * For a fixed K, Choice h is affine in p_max.
4. **Checkpoint selection.** Upstream keeps the best-dev checkpoint, which may be earlier and less
   memorized than the final one. `run_meta.json` records `best_step`.
5. **Fixed y under counterfactuals.** Replacing F1 changes the correct route, and the replacement topic
   may not even be in A. y stays fixed, as specified; per-grid `rule_gold_changes_*` flags are saved for analysis.
6. **Evaluation-only subsets.** The confidence-matched and saturated subsets use target labels to build
   the subset. They are analyses, not attacks.
7. **What "seeds" means.** Feasibility seeds vary SFT randomness on one dataset. Data-seed variation means
   generating extra datasets with other `--seed` values.
8. **Novelty.** Perturbation- and neighbourhood-based MIAs already exist for language models (for example
   neighbourhood comparison, Mattern et al., 2023). Do a proper literature review before claiming novelty for RIFT.
9. **RLCD (Milestone 8, not implemented).** NanoJev already provides an RLCD-*inspired* sampled
   proper-reward loss (`--loss paired_brier_pg --objective observed_outcome`, `docs/RLCD_EXPERIMENT.md`),
   which can warm-start from an SFT checkpoint via `--init-checkpoint`. `sft_command()` already accepts
   `init_checkpoint`. That loss is not TypeSafe's recipe, and it needs rows with
   `gold_label_kind: observed_outcome`. Eve-RLCD (`anthony-maio/eve-rlcd`) is the alternative.
   Keep SFT-only and SFT+RLCD results separate.
10. **The CPU shim** only redirects upstream's CUDA calls for smoke tests. It refuses bf16 and must never
    produce reported results.
