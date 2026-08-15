# Stage 2 — Architecture Improvement & Results

*Improvement half of the project. Self-contained in `stage2/`; the Stage 1
reconstruction code (`covidgan/`, `train_classifier.py`, …) is unchanged and only
imported for its data/metrics utilities. Curated numbers behind every table live
in `stage2/results/`.*

---

## 1. What we changed

Two coupled, standard modifications to the paper's VGG16 detector:

| # | Change | Type (assignment list) | Where |
|---|---|---|---|
| 1 | **Unfreeze the top VGG16 conv block(s)** — domain fine-tuning of the encoder | "Change the encoder" | `stage2/model.py` |
| 2 | **Add BatchNorm to the head** (`Dense(64) → BatchNorm1d → ReLU → Dropout → Dense(2)`) | "Add normalization layers" | `stage2/model.py` |

Supporting recipe: **discriminative learning rates** — the fresh head trains at
`1e-3`, the unfrozen (pretrained) backbone at `1e-5`, so ImageNet filters are
gently adapted rather than wiped out on ~900 images.

We report **two variants** to show the effect of how much encoder capacity we
unlock:

| Variant | Top blocks fine-tuned | Trainable params |
|---|---|---|
| Stage 1 (baseline) | 0 (frozen) | 33K |
| **Stage 2 / uf=1** | 1 | 7.11M |
| **Stage 2 / uf=2** | 2 | 13.01M |

## 2. Why we expected it to improve results

A **direct consequence of the Stage 1 analysis** (`FINDINGS.md` §8.3), not a guess.
Stage 1 established that our frozen-backbone reconstruction hit ~90.6% and that GAN
augmentation was **flat** — the decisive evidence being that FID *halved*
(504 → 273) yet CNN-SA did not move, isolating the blocker as the **frozen head**
(synthetic data could nudge only a ~33K linear boundary), **not** image quality.

The fix is to **create capacity where augmentation can act**: unfreeze the top
block(s) so the encoder can (a) adapt ImageNet features to chest-X-ray texture —
raising the baseline — and (b) actually be reshaped by the extra synthetic
examples — reviving the augmentation effect the paper reports.

## 3. Results — the improved model (full data, 5 seeds)

Test set: the **same 192 real images** (72 COVID + 120 Normal) used throughout.
Stage 2 rows are the **mean ± std over seeds {0,1,2,3,4}** (`stage2/multiseed.py`,
`stage2/results/multiseed_uf{1,2}.json`); paper and Stage 1 are their reported
single runs.

| Model | CNN-AD (real only) | CNN-SA (+ synthetic) | Augmentation lift |
|---|---|---|---|
| **Paper** (Waheed et al. 2020) | 85.00% | 95.00% | **+10.00 pts** |
| **Stage 1 reconstruction** (frozen VGG16) | 90.62% | 90.10% | **−0.52 pts** (flat) |
| **Stage 2 / uf=1** (1 block + BN) | 92.50% ± 0.71 | 93.65% ± 1.16 | **+1.15 pts** |
| **Stage 2 / uf=2** (2 blocks + BN) | **94.48% ± 0.97** | **96.67% ± 0.71** | **+2.19 pts** |

COVID recall (where the whole accuracy gap lives — FINDINGS §3):

| Model | CNN-AD COVID recall | CNN-SA COVID recall | Recall lift |
|---|---|---|---|
| Stage 2 / uf=1 | 88.89% ± 1.76 | 91.39% ± 1.62 | +2.50 pts |
| Stage 2 / uf=2 | 93.33% ± 3.22 | 96.67% ± 2.08 | +3.33 pts |

**Reading.** More unlocked capacity helps on both axes: uf=2 raises the baseline
(90.6 → 94.5%) *and* roughly doubles the augmentation lift vs uf=1. At uf=2,
**CNN-SA (96.67%) exceeds the paper's 95%**, and the augmentation effect is
consistent across all 5 seeds. std = population std over seeds.

## 4. Supporting analysis A — data-scarcity curve

*(`stage2/data_scarcity.py`, uf=2, 3 seeds/point, `stage2/results/scarcity_uf2.json`.)*

Why is our full-data lift (~+2 pts) smaller than the paper's +10? Because our
baseline is already near ceiling on the clean modern data. The paper's +10 came
from a **data-starved 85% baseline**. To test that directly we subsample the
**real** training set (stratified) while keeping the **full** synthetic pool fixed:

| real frac | n_real | CNN-AD | CNN-SA | acc lift | recall lift |
|---|---|---|---|---|---|
| 0.10 | 93 | 86.98% | 89.24% | +2.26 | +1.39 |
| 0.25 | 233 | 89.58% | 92.88% | **+3.30** | +4.63 |
| 0.50 | 466 | 91.32% | 93.06% | +1.74 | +0.93 |
| 1.00 | 932 | 93.92% | 96.01% | +2.08 | +1.39 |

**Honest reading.** Two robust facts: (1) the CNN-AD baseline degrades
monotonically as real data shrinks (93.9 → 87.0%), as expected; (2) **GAN
augmentation helps at *every* data level** (+1.7 to +3.3 pts). The effect is
**largest in the low-to-mid regime (peak +3.30 at 25%)**, supporting the paper's
"augmentation rescues data-starved models" thesis *directionally*. It is **not** a
clean monotonic curve, and the extreme-scarce 10% point is **noisy** (only 93
images; its three seeds ran +4.17 / −1.04 / +3.65), so we frame it as
"consistently positive, strongest at low-to-mid data," not "monotonically growing."

## 5. Supporting analysis B — train/test generalization curve (diagnostic)

*(`stage2/diagnostic_curve.py`, uf=2, seed 0, 25 epochs,
`stage2/results/diagnostic_curve_{ad,sa}.csv`.)*

**Methodology note.** The reported models train for a **fixed 15 epochs chosen
test-blind** (we never looked at test to pick it), which is unbiased. This
diagnostic *additionally* evaluates the test set every epoch **for analysis only**
— it is **not** used to select the stopping epoch (doing so would be selecting on
the test set and would bias the result).

Findings:

- **Train saturates early but test does NOT degrade.** Train accuracy reaches
  ≥99% by epoch 6 (AD) / epoch 3 (SA), yet through epoch 25 the test accuracy
  never falls: AD sits on a flat plateau ≈ 93.7% (±0.76), SA plateaus ≈ 96.5% and
  even drifts slightly upward. So "just train longer" neither helps much nor hurts
  — there is **no harmful overfitting**, only a plateau.
- **Fixed 15 epochs lands on the plateau**, confirming the choice was reasonable.
- **SA's test curve sits above AD's at essentially every epoch**, so the
  augmentation benefit is persistent, not an artifact of the epoch-15 snapshot.
- The single-epoch "peaks" (AD 95.83% @ ep24, SA 97.40% @ ep6) are random noise
  spikes at different epochs — a concrete illustration of *why* picking the epoch
  by test score would be biased.

**Conclusion.** Early stopping is not needed for a valid result here; the
test-blind fixed epoch count is unbiased and sits on the generalization plateau.
Proper validation-based early stopping (stopping on a held-out *validation* split,
never the test set) is noted as a legitimate future refinement.

## 6. Discussion — what worked, what didn't, what we learned

**Worked:**
1. **Higher baseline.** Fine-tuning the encoder beat fixed ImageNet features
   (90.6 → 94.5% at uf=2).
2. **The paper's core claim is restored.** With headroom present, augmentation
   *helps* — consistently across seeds, larger with more unlocked capacity, and
   largest in the scarce-data regime. The COVID-recall lift exceeds each arm's
   seed spread, so it is a real effect, not noise.

**Did not reproduce:** the paper's **+10-point** magnitude. We recover the
*direction and mechanism* of the claim but not its size — our starting point
(~94.5% on cleaner modern data) simply has far less room than the paper's 85%.

**Learned.** The CovidGAN augmentation benefit is **real but contingent on model
capacity and baseline headroom**: it appears when the classifier can adapt its
features and is not already at ceiling, and vanishes when the backbone is frozen
or the baseline is already high. Stage 1 (frozen) and Stage 2 (fine-tuned, uf=1/2)
on the *same data and synthetic pool* bracket both regimes and make the mechanism
explicit; the data-scarcity curve shows the effect grows as the baseline weakens.

**Limits.** Numbers are 5-seed (full data) / 3-seed (scarcity) means; the 10%
scarcity point is noisy. The open Stage 1 caveat (FINDINGS §7 — a possible
Kaggle-specific fine-scale artifact, testable only with an independent dataset)
applies here too.

## 7. How to reproduce

```bash
# improved model, both arms (uf=2 shown; use --unfreeze-blocks 1 for uf=1)
python -m stage2.train_stage2 --mode ad --unfreeze-blocks 2 --out-dir runs/stage2_cnn_ad
python -m stage2.train_stage2 --mode sa --unfreeze-blocks 2 --synthetic-dir data/synthetic \
    --out-dir runs/stage2_cnn_sa

# multi-seed AD-vs-SA (mean ± std) — run for uf=1 and uf=2
python -m stage2.multiseed --seeds 0 1 2 3 4 --unfreeze-blocks 1 --out-root runs/stage2_multiseed
python -m stage2.multiseed --seeds 0 1 2 3 4 --unfreeze-blocks 2 --out-root runs/stage2_multiseed_uf2

# supporting analyses
python -m stage2.data_scarcity --fractions 0.1 0.25 0.5 1.0 --seeds 0 1 2 --unfreeze-blocks 2
python -m stage2.diagnostic_curve --unfreeze-blocks 2 --epochs 25   # diagnostic only

# assemble the paper / reconstruction / improved table
python -m stage2.compare_results
```

Ablations: `--unfreeze-blocks {0..5}` (0 = Stage 1 behaviour + BN head);
`--no-head-bn` drops the BatchNorm.
