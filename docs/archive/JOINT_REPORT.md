# Reproducing, Diagnosing, and Rescuing the CovidGAN Claim — Validated on a Second Dataset

**Final Project, Part 2 — Joint Report.**
Reconstruction, diagnosis, and improvement of Waheed et al.,
*"CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved
Covid-19 Detection,"* IEEE Access, vol. 8, 2020.

**Contributors**
- **Hila Fishman** — the primary CovidGAN research (`CovidGAN-Pytorch`):
  reconstruction, the diagnostic investigation, and both improvement axes on
  COVID-CXR (de-fingerprinting the GAN, and unfreezing classifier capacity). This
  is the main line of the project and leads every experiment below.
- **Maya Hayat** — the second-dataset replication (`DDSM-ACGAN-Pytorch`): running
  the same pipeline and the same experiments on CBIS-DDSM mammography, to test
  whether the CovidGAN conclusions are specific to our COVID dataset or hold on
  independent data.

Long-form working logs remain in the repos as appendices: CovidGAN — `REPORT.md`,
`FINDINGS.md`, `SESSION_FINDINGS.md`, `stage2/STAGE2_FINDINGS.md`; DDSM —
`FINDINGS.md`, `STAGE2_FINDINGS.md`, `README.md`.

> **How to read this report.** CovidGAN is the primary dataset and **leads every
> experiment**. Each main experiment states the COVID result first, then a
> **"Second dataset (DDSM)"** paragraph that reports whether the same conclusion
> holds on independent data. Boxes marked **⚠️ COVID GAP** flag places where the
> second dataset currently carries a test that the primary COVID dataset does not
> yet lead (missing, or run at lower statistical power) — i.e. work to bring back
> onto the main line.

---

## 0. Overview and flow of the project

**CovidGAN is the main project.** We reconstruct the paper's AC-GAN + VGG16
COVID-19 detector faithfully in PyTorch, and find that its **central claim — a
+10-point accuracy lift from GAN augmentation — does not reproduce** (§3).

Our first hypothesis for *why* was the **data**: the modern Kaggle CXR set we use
is cleaner and more separable than the paper's 2020 hand-merged set, which could
give a baseline so high that augmentation has nothing left to add. If the failure to
reproduce were really an artifact of our particular dataset, it should behave
differently on a **different** dataset. So the project runs on two tracks:

1. **Diagnose and improve on the COVID data (the main line, §3–§9).** A sequence of
   controlled experiments isolates *why* the claim fails and designs two fixes — a
   **GAN that produces transferable pathology instead of a class fingerprint**, and
   an **unfrozen classifier with capacity to use it**.
2. **Replicate on a second dataset (DDSM, woven through §4–§8).** We re-run the same
   experiments on CBIS-DDSM benign-vs-malignant mammography — an independent, static
   benchmark — to test whether each COVID conclusion is data-specific or general.

The headline: **the paper's naive setup fails on both datasets, the same fixes
rescue a real (smaller, contingent) benefit on both, and both datasets independently
reveal that a generator better by every quality metric can make the classifier
worse.** §10 sets out how to fuse the two tracks into one controlled experiment.

---

## 1. Shared architecture

Both datasets use the identical model family — a conditional AC-GAN plus a VGG16
classifier head — so a difference in results is attributable to the data, not the
model. Full detail in `CovidGAN-Pytorch/REPORT.md` §1; summary:

- **AC-GAN generator:** class label + noise `z ∈ ℝ¹⁰⁰` → branches concatenated to
  `7×7×1025` → four stride-2 transpose-conv blocks (`7→14→28→56→112`) → `Tanh` →
  `112×112×3` (~22M params).
- **AC-GAN discriminator:** five conv blocks → `7×7×512` → **two heads** (validity +
  class) (~2M params).
- **Classifier:** frozen ImageNet VGG16 → `GAP → Dense(64) → Dropout → Dense(2)`;
  only the ~33K-param head trains, per the paper.
- **Objective:** `BCEWithLogits` (validity) + `CrossEntropy` (class) for the GAN;
  `CrossEntropy` for the classifier. GAN: batch 64, Adam `lr=2e-4`, `β₁=0.5`.

Two **improvement axes** recur: **GAN architecture** (make synthetic carry
transferable signal, not a fingerprint) and **classifier capacity** (unfreeze the
top VGG16 blocks + BatchNorm head so the classifier can use it).

---

## 2. The paper's claim (the target)

Accuracy on a held-out **192-image** real COVID-CXR test set (72 COVID + 120
Normal), in two configurations reused on both datasets:

- **CNN-AD** — trained on **real** data only.
- **CNN-SA** — trained on **real + GAN-synthesized** data.

| Configuration | Accuracy |
|---|---|
| CNN-AD (real only) | **85%** |
| CNN-SA (+ synthetic) | **95%** |

The claim under test is the **+10-point** lift (CNN-SA − CNN-AD), concentrated in
COVID recall.

**Metric note.** *Accuracy* = correct/total on real test. *Recall* = TP/(TP+FN) per
class. *FID* judges *generator* realism (lower = better). A *transfer probe* (train
on synthetic alone, test on real) judges whether synthetic carries real class
signal. A recurring finding: **none of these generator-side measures predicts
downstream augmentation value.**

---

## 3. Reconstruction: the paper's claim does not reproduce, and why we added a second dataset

We trained the GAN for the paper's **2000 epochs**, generated a synthetic pool
(1669 COVID + 1399 Normal), and evaluated on the 192-image real test set.

| Configuration | Paper | Our reconstruction |
|---|---|---|
| CNN-AD (real only) | 85.00% | **90.62%** |
| CNN-SA (+ synthetic) | 95.00% | **90.10%** |
| Augmentation lift | +10.00 | **−0.52 (flat)** |

Two anomalies: (A) our baseline is *higher* than the paper's; (B) augmentation is
**flat**, not +10. Controlled tests (in `FINDINGS.md`) rule out a bug (layer audit
matches; VGG16 frozen), a cross-source shortcut (single-source split still 91.15%),
and a coarse shortcut (destroying fine detail decays accuracy 89.6→74.0% and halves
COVID recall — the model is detail-dependent).

**The working hypothesis this raised — and the reason for DDSM.** The most likely
explanation for the high, augmentation-proof baseline is that our **modern dataset
is cleaner and more separable** than the paper's, so our detector does legitimate
but easier classification and the +10 has no room to appear. That is a claim about
*our data*, not about the method — so the way to test it is to **run the whole
pipeline again on a completely different dataset**. If the same conclusions recur on
independent data, they are not an artifact of the COVID set. That second dataset is
**CBIS-DDSM** benign-vs-malignant mammography (Maya Hayat) — a static, versioned
benchmark whose labels and train/test split come from its own case CSVs (not
re-randomized). Mass-only scope: **1,318 real training ROIs, 378 test (231 benign /
147 malignant)**; DDSM numbers are from Weights & Biases
(`maya-hayat-ariel-university/ddsm-acgan`). Two DDSM data bugs were fixed before any
result below — **mask-folder contamination** and **16-bit `I;16` grayscale
truncation**.

The rest of the report runs the **main experiments on COVID** and checks each on
**DDSM as the second dataset.**

---

## 4. Experiment 1 — Is the synthetic pool transferable, or a class fingerprint?

**COVID (leads).** Train a classifier on the synthetic pool alone, test on real
(floor = 62.5%). The Stage-1 2000-epoch pool scored **55.2%** — *below* floor (COVID
recall 0.22): the synthetic class signal is a **fingerprint**, not pathology.
Quantified: the GAN made COVID samples uniformly hazy/bright and Normal dark/sharp,
but on *real* data class-mean brightness differs by only **+8.8 / 255 (~3.5%)** and
per-image ranges overlap almost completely — the GAN amplified a faint, overlapping
tendency into a deterministic rule. Root cause (ranked): `noise_std=0.02` (with
`z≈0` the generator collapses to one prototype per class, separable only by global
style), the AC-GAN auxiliary objective rewarding any class separability,
discriminator overfitting on ~400 COVID images, and 2000 epochs sharpening it.

**Second dataset (DDSM): holds.** DDSM's own synthetic-only probe likewise shows the
naive pool carries little transferable signal, and its PCA analysis shows baseline-GAN
synthetic malignant samples sitting *outside* the real-malignant feature region —
the same "synthetic is off in its own corner of feature space" failure, independently
reproduced.

---

## 5. Experiment 2 — Improving the GAN so synthetic carries real signal

**COVID (leads).** Fixes: raise `noise_std` 0.02→**1.0** (breaks prototype collapse),
add **DiffAugment** + **spectral norm** (small-data stability), and replace the
auxiliary-classifier conditioning with a **projection discriminator** (removes the
fingerprint incentive). Effect on generator quality and transfer:

| Generator | FID overall | Transfer-probe accuracy |
|---|---|---|
| Stage-1 AC-GAN, 2000 ep (noise 0.02) | 272.7 | **55.2%** (below floor) |
| Improved AC-GAN, 300 ep | 225.7 | 72.9% |
| **Projection discriminator, 300 ep** | 155.0 | 74.5% |
| **Projection discriminator, 600 ep** | **121.3** | **81.3%** (COVID recall 0.47→0.67) |

Transfer moved 55%→81%; the projection generator at 300 ep already beats the Stage-1
AC-GAN at 2000 ep by ~43% FID.

**Second dataset (DDSM): holds (different recipe).** DDSM improved its GAN too —
**spectral normalization** on the discriminator + a **kernel 5→4 / stride 2** fix
(removing checkerboard artifacts) — cutting loss variance **~10–30×** (D_loss std
0.351→0.026) and moving synthetic malignant samples into overlap with the
real-malignant region in PCA. Same conclusion (a de-fingerprinted, more stable GAN is
achievable), reached via a **different intervention** — the two recipes are not yet
unified (§10).

---

## 6. Experiment 3 — Does a better generator convert to a better classifier?

This is the project's central question, and its most surprising answer.

**COVID (leads).** Taking the projection pool from 300→600 epochs (FID 155→121,
transfer 74.5→81.3% — better by *every* generator-side metric) moved downstream
accuracy the **wrong way** in both regimes (single seed, only the pool changing):

| Classifier | AD (real only) | SA + Proj 300 | SA + Proj 600 |
|---|---|---|---|
| frozen | 90.62% | 90.10% | **88.54%** |
| unfrozen (uf=2) | 95.31% | 97.40% | **93.75%** |

Lower FID *and* higher transfer, yet downstream dropped — **the sharpest evidence
that GAN quality metrics do not predict downstream augmentation value.** (Also
consistent: earlier, FID halving from the 25-ep to the 2000-ep generator produced
zero downstream change.)

**Second dataset (DDSM): holds — same reversal.** DDSM's improved GAN *helps* at
matched epochs (baseline-GAN SA 60.05% → improved-GAN SA 66.93%, +6.9, real-only AD
64.02%) **but hurts once the classifier is unfrozen** (unfrozen SA-improved 66.67% <
unfrozen SA-baseline 70.77% and < unfrozen AD 69.45%, at both seeds). The same
counter-intuitive phenomenon — a better-by-metrics generator degrading a
high-capacity classifier — appears independently on both datasets.

> **⚠️ COVID GAP — this experiment is currently *led* by DDSM, not COVID.**
> On DDSM the reversal is established at **two seeds** and its mechanism was probed
> directly (a synthetic-only fingerprint test that *rejected* the "learnable
> fingerprint" explanation — the improved pool carries *more* transferable signal,
> not less). On COVID the same reversal is only a **single-seed** observation
> (projection-600) with **no mechanism probe**. To keep COVID leading, the
> projection-300-vs-600 downstream comparison needs multi-seed runs, and a
> COVID synthetic-only probe of the 300 vs 600 pools to test the mechanism.

---

## 7. Experiment 4 — Unfreezing classifier capacity

**COVID (leads).** Fine-tune the top VGG16 conv block(s) (trainable params 33K →
7.11M at uf=1 → 13.01M at uf=2) + a BatchNorm head, discriminative LRs (head `1e-3`,
backbone `1e-5`). Mean ± std over **5 seeds**:

| Model | CNN-AD | CNN-SA | Lift |
|---|---|---|---|
| **Paper** | 85.00% | 95.00% | +10.00 |
| **Reconstruction** (frozen) | 90.62% | 90.10% | −0.52 (flat) |
| **Stage 2 / uf=1** | 92.50 ± 0.71 | 93.65 ± 1.16 | **+1.15** |
| **Stage 2 / uf=2** | **94.48 ± 0.97** | **96.67 ± 0.71** | **+2.19** |

Unfreezing raised the baseline (90.6→94.5%) *and* revived augmentation; at uf=2,
**CNN-SA 96.67% exceeds the paper's 95%.** The COVID-recall lift (+3.33) exceeds each
arm's seed spread.

**Second dataset (DDSM): holds.** Unfreezing raised the DDSM baseline identically —
frozen AD 66.00% → unfrozen AD 69.45% (+3.45), consistent at both seeds — and with
the unfrozen classifier, baseline-GAN augmentation gives DDSM's **first fully
reproducible augmentation win** (70.77% > 69.45% at both seeds). Capacity is the
precondition for augmentation to help, on both datasets.

---

## 8. Experiment 5 — Both fixes are required (capacity × GAN-source grid)

**COVID (leads).** Crossing {improved AC-GAN, projection} pools × {frozen, unfrozen}
classifier (single seed; the uf=2 lift separately confirmed multi-seed at +2.19 ±
0.7):

| Classifier | AD | SA + AC-GAN | SA + Projection |
|---|---|---|---|
| frozen | 90.62% | 90.10% | 90.10% |
| unfrozen | 95.31% | **97.92%** | 97.40% |

Better GANs help **only when the encoder is unfrozen**; under the frozen head they
stay flat, and with **scarce** real data the frozen head makes augmentation actively
*hurt* (acc lift −1.9 at 100% real → −8.2 at 10%). Neither fix alone reproduces the
paper's benefit — the GAN improvement and the capacity improvement are
**complementary and both necessary**.

**Second dataset (DDSM): holds, more rigorously.** DDSM ran the analogous grid —
{frozen, unfrozen} × {real, baseline-GAN, improved-GAN} — at **two seeds**, and finds
the same structure: augmentation is flat under the frozen head and only becomes a
reproducible win once the classifier is unfrozen.

> **⚠️ COVID GAP — the second dataset carries this grid at higher power.**
> COVID's capacity × GAN-source grid above is **single-seed**; DDSM's is **two-seed**
> (§5.3 numbers). For the primary dataset to lead, the COVID grid should be re-run
> multi-seed via `stage2/multiseed.py` so the frozen/unfrozen × GAN-source cells
> carry error bars comparable to DDSM's.

---

## 9. Experiment 6 — Data-scarcity sweep (COVID only)

**COVID (leads).** Subsampling real data 10/25/50/100% at a fixed synthetic pool
(unfrozen, uf=2, 3 seeds): the augmentation lift is positive at every level and
largest in the low-to-mid regime — +2.26 (10%), **+3.30 (25%)**, +1.74 (50%), +2.08
(100%). This supports the paper's "augmentation rescues data-starved models" reading
and explains why our full-data lift is ~+2 not +10: our baseline starts near ceiling,
the paper's started at a starved 85%.

*This experiment was not run on DDSM* — so there is no second-dataset check of the
scarcity conclusion. (Noted as future work in §12; not a COVID gap, since COVID
leads it.)

---

## 10. Stage 2 — combining the two datasets into one experiment

Both improvement axes have now been run on **both** datasets, but the two tracks are
**parallel, not yet a single controlled experiment** — the GAN recipes differ and the
statistical power differs:

| Improvement axis | CovidGAN (COVID-CXR) | DDSM (mammography) |
|---|---|---|
| **GAN architecture** | noise 1.0 + DiffAugment + spectral norm + **projection discriminator** (§5); mostly single-seed | spectral norm + **kernel/stride** (§5); mostly single-run |
| **Classifier capacity** | unfreeze VGG + BN head (§7); **5-seed** | unfreeze VGG + BN head (§7); **2-seed** |

**Plan to fuse the tracks:**
1. **Unify the improved-GAN recipe.** Run the *same* generator improvement on both —
   most naturally, port the **projection + noise-1.0** recipe
   (`stage2/gan_projection.py`) onto DDSM, so "improved GAN" means the same thing on
   both datasets.
2. **Run one shared grid on both:** `{baseline GAN, improved GAN} × {frozen,
   unfrozen}`, **multi-seed**, reporting AD-vs-SA on each test set. This directly
   closes the two ⚠️ COVID GAPs (§6, §8).
3. **Nail the shared reversal.** The "better GAN → worse downstream" effect (§6)
   appears on both datasets but is under-powered on each; the multi-seed grid
   confirms whether it is real, and a real:synthetic ratio sweep tests the
   interaction/dilution mechanism.

The deliverable is one table — **two datasets × two GAN qualities × two classifier
capacities, matched protocol, error bars** — stating the unified claim quantitatively.

---

## 11. Discussion

**The paper's claim reproduces in kind, not in magnitude, and only under
conditions.** The naive +10-point setup failed on **both** datasets — and crucially,
it failed on DDSM too, so the failure is **not an artifact of our cleaner COVID
dataset** (the §3 hypothesis that motivated adding DDSM). The same family of fixes
rescued a real but smaller benefit on each: +2.19 on COVID (uf=2), and on DDSM the
first reproducible win plus a +6.9 matched-epoch GAN-architecture gain.

**Generation quality does not predict downstream benefit — the strongest
cross-dataset finding.** On COVID, FID halved and the transfer probe rose 55→81% with
*zero or negative* downstream change. On DDSM, a 10–30× more stable, better-aligned
generator helped at matched epochs yet hurt the high-capacity classifier. Both
datasets independently produced the same reversal. Only an AD-vs-SA comparison
measures the paper's actual claim.

**Two knobs govern whether augmentation helps.** *Capacity:* unfreezing the encoder
raised the baseline on both tasks and is the precondition for any augmentation gain.
*Headroom:* the paper's +10 is a data-starved, low-baseline regime — reproducible in
kind but not in magnitude on data (either dataset) that is not starved.

---

## 12. Limitations & future work

- **Close the two ⚠️ COVID GAPs (§6, §8):** multi-seed the COVID projection-300-vs-600
  downstream comparison and the capacity × GAN-source grid, so the primary dataset
  leads at the power DDSM already has.
- **Combine into one matrix (§10):** unify the improved-GAN recipe across datasets and
  run the shared multi-seed grid on both — the single most valuable remaining
  experiment.
- **Replicate the scarcity sweep (§9) on DDSM** for a second-dataset check.
- **Explain the reversal:** vary the real:synthetic ratio (vs. fixed mixes) to test
  interaction/dilution.
- **Tame the frozen-head penalty:** cap synthetic to ~1× real, or augment only the
  minority class — untested levers in `SESSION_FINDINGS.md`.
- **A fine-scale COVID artifact is not fully ruled out** — only an independent public
  CXR dataset would settle it. Validation-based early stopping is a further refinement.

---

## 13. References

1. A. Waheed, M. Goyal, D. Gupta, A. Khanna, F. Al-Turjman, P. R. Pinheiro,
   "CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved
   Covid-19 Detection," *IEEE Access*, vol. 8, pp. 91916–91923, 2020.
2. A. Odena, C. Olah, J. Shlens, "Conditional Image Synthesis with Auxiliary
   Classifier GANs," *ICML*, 2017.
3. T. Miyato, M. Koyama, "cGANs with Projection Discriminator," *ICLR*, 2018.
4. K. Simonyan, A. Zisserman, "Very Deep Convolutional Networks for Large-Scale
   Image Recognition" (VGG16), *ICLR*, 2015.
5. M. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a
   Local Nash Equilibrium" (FID), *NeurIPS*, 2017.
6. T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida, "Spectral Normalization for
   Generative Adversarial Networks," *ICLR*, 2018.
7. A. Odena, V. Dumoulin, C. Olah, "Deconvolution and Checkerboard Artifacts,"
   *Distill*, 2016.
8. S. Zhao, Z. Liu, J. Lin, J.-Y. Zhu, S. Han, "Differentiable Augmentation for
   Data-Efficient GAN Training" (DiffAugment), *NeurIPS*, 2020.
9. T. Rahman, M. E. H. Chowdhury et al., "COVID-19 Radiography Database," Kaggle.
10. R. S. Lee, F. Gimenez, A. Hsu et al., "A curated mammography data set for use in
    computer-aided detection and diagnosis research" (CBIS-DDSM), *Scientific Data*,
    2017.

---

*Code: CovidGAN in `CovidGAN-Pytorch/` (`covidgan/`, `train_gan.py`,
`train_classifier.py`, `generate_synthetic.py`, `evaluate_fid.py`, `stage2/`
including `gan_projection.py`, `synthetic_only_probe.py`). DDSM in
`DDSM-ACGAN-Pytorch/` (`ddsm_acgan/`, `prepare_ddsm.py`, `train_gan.py`,
`generate_synthetic.py`, `train_classifier.py`, `multiseed.py`). Both are subfolders
of the `generative_models` monorepo.*
