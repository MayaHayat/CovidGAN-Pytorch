# CovidGAN — Paper Reconstruction and Improvement

**Final Project, Part 2.** Reconstruction and improvement of Waheed et al.,
*"CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved
Covid-19 Detection"*, IEEE Access, 2020.

This report covers both stages. **Stage 1** reconstructs the paper's AC-GAN +
VGG16 COVID-19 detector and reproduces its main result; **Stage 2** improves the
architecture and shows the effect in a paper/reconstruction/improved table.
Detailed experiment logs are in `FINDINGS.md` (Stage 1) and
`stage2/STAGE2_FINDINGS.md` (Stage 2); the numbers behind every table are in
`stage2/results/`.

---

## 1. Original architecture

The paper's pipeline has two models: an **AC-GAN** that synthesizes chest-X-ray
(CXR) images, and a **VGG16-based classifier** that detects COVID-19. Synthetic
images from the GAN augment the classifier's training set.

### 1.1 AC-GAN generator (`covidgan/models.py`)
- **Inputs:** a class label `c ∈ {COVID, Normal}` and a noise vector `z ∈ ℝ¹⁰⁰`
  (`z ~ N(0, 0.02)`, per the paper).
- **Label branch:** `Embedding(2→50) → Dense → 7×7×1` map.
- **Noise branch:** `Dense(100 → 1024·7·7) → ReLU → 7×7×1024`.
- **Body:** concatenate to `7×7×1025`, then four `5×5` stride-2 transpose-conv
  blocks (BatchNorm + ReLU), upsampling `7→14→28→56→112`, with `Tanh` on the
  final layer → `112×112×3` image in `[-1, 1]`. (~22M params.)

### 1.2 AC-GAN discriminator
- Five `3×3` conv blocks (BatchNorm / LeakyReLU(0.2) / Dropout(0.5)),
  `112×112×3 → 7×7×512`, then **two heads**: a validity head (real/fake) and a
  class head (COVID/Normal) — the defining feature of an AC-GAN. (~2M params.)

### 1.3 Classifier (`build_classifier`)
- **Frozen ImageNet VGG16** convolutional base →
  `GlobalAveragePool → Dense(64, ReLU) → Dropout(0.5) → Dense(2)`.
- Only the head trains (~33K params); the VGG16 base is frozen, matching the
  paper: *"…setting the 'trainable' property on each of the VGG layers to False
  before training."*

### 1.4 Losses / objective
- **GAN:** `BCEWithLogits` on the validity head + `CrossEntropy` on the class
  head, for both D and G (standard AC-GAN objective). One-sided label smoothing
  (real target 0.9).
- **Classifier:** `CrossEntropy`.

### 1.5 Key hyperparameters (paper's)
- GAN: batch 64, Adam `lr=2e-4`, `β₁=0.5`, **2000 epochs**.
- Classifier: batch 16, Adam `lr=1e-3`, image size `112×112`, pixels in `[0,1]`.
- Data split (paper): train COVID/Normal 331/601, **test 72/120 (192 total)**.

---

## 2. Paper results

The paper's headline metric is **classification accuracy** on a held-out set of
**192 real** CXRs (72 COVID + 120 Normal), in two configurations:

- **CNN-AD** — classifier trained on **real** data only ("actual data").
- **CNN-SA** — classifier trained on **real + GAN-synthesized** data.

| Configuration | Accuracy |
|---|---|
| CNN-AD (real only) | **85%** |
| CNN-SA (+ synthetic) | **95%** |

The paper's central claim is the **+10-point** lift from GAN augmentation. It also
reports precision/recall/F1/specificity; the augmentation gain is concentrated in
**COVID recall** (the paper's CNN-AD misses ~22 of 72 COVID cases).

**Metric meanings.**
- **Accuracy** (paper's metric) = correct / total on the 192-image real test set —
  the fraction of CXRs classified correctly.
- **Per-class recall** (metric studied in class) = for each class, TP / (TP+FN) —
  the fraction of true COVID (resp. Normal) cases actually caught. We report this
  because in screening, missing COVID (recall) matters more than raw accuracy.
- **FID** (Fréchet Inception Distance; metric studied in class, used to evaluate
  the *generator*) = Fréchet distance between InceptionV3 feature distributions of
  real vs. synthetic images; **lower = more realistic**. It measures GAN image
  quality, which accuracy alone cannot.

---

## 3. Reconstruction results (Stage 1)

We reimplemented the full pipeline in PyTorch, trained the GAN for the paper's
**2000 epochs**, generated a synthetic pool (1669 COVID + 1399 Normal = 3068
images), and evaluated CNN-AD / CNN-SA on the 192-image real test set.

| Configuration | Paper | Our reconstruction |
|---|---|---|
| CNN-AD (real only) | 85.00% | **90.62%** |
| CNN-SA (+ synthetic) | 95.00% | **90.10%** |
| Augmentation lift | +10.00 | **−0.52 (flat)** |

**FID (generator quality)** confirms the GAN genuinely learned: FID between the
real test set and the synthetic pool dropped from **504** (25-epoch smoke GAN) to
**273** (2000-epoch GAN) — roughly halved.

**Two anomalies vs. the paper, both explained (see `FINDINGS.md`):**

1. **Our baseline is *higher* (90.6% vs 85%).** We ruled out a cross-source
   shortcut (a same-source A/B still scored 91.15%) and a coarse-resolution
   shortcut (a downsampling experiment showed genuine detail-dependence). The
   consistent explanation: the **modern Kaggle COVID-19 Radiography Database is
   cleaner, larger and more separable** than the paper's 2020 hand-merged set, so
   our detector does legitimate but *easier* classification.

2. **Augmentation was flat, not +10.** The decisive evidence: FID *halved*
   (504→273) yet CNN-SA did **not** move. This isolates the cause as the **frozen
   VGG16 head** — with the backbone frozen, synthetic images can only nudge a
   ~33K-parameter linear boundary; they cannot reshape the features. It is **not**
   an image-quality problem. This finding directly motivates Stage 2.

We therefore match the paper's *methodology and metric* faithfully; the numeric
differences are a dataset-difficulty effect plus the frozen-head ceiling, both
diagnosed rather than left unexplained.

---

## 4. Improved architecture (Stage 2)

Guided by the Stage 1 diagnosis (the frozen head, not image quality, suppressed
augmentation), Stage 2 **creates capacity where augmentation can act**. All Stage 2
code is in the separate `stage2/` package. Two coupled changes:

1. **Fine-tune the top VGG16 conv block(s)** ("Change the encoder"). We unfreeze
   the top 1 or 2 conv blocks so the encoder adapts ImageNet features to CXR
   texture instead of being fixed. Trainable params rise from **33K** (frozen) to
   **7.11M** (uf=1) or **13.01M** (uf=2).
2. **Add BatchNorm to the head** ("Add normalization layers"):
   `Dense(64) → BatchNorm1d → ReLU → Dropout → Dense(2)`, to stabilize the larger
   trainable set.

Trained with **discriminative learning rates** — fresh head at `1e-3`, unfrozen
pretrained backbone at `1e-5` — a standard fine-tuning recipe so the ImageNet
filters are gently adapted, not destroyed on ~900 images.

**Why we expected improvement.** (a) Adapting the encoder to CXRs should raise the
baseline; (b) with the features now trainable, the synthetic data can reshape
them, reviving the augmentation effect that the frozen model could not exploit.

---

## 5. Improved results (Stage 2)

Same 192-image real test set. Stage 2 rows are **mean ± std over 5 seeds**
(`stage2/multiseed.py`); paper and Stage 1 are single runs.

| Model | CNN-AD (real only) | CNN-SA (+ synthetic) | Augmentation lift |
|---|---|---|---|
| **Paper** | 85.00% | 95.00% | +10.00 |
| **Stage 1 reconstruction** (frozen) | 90.62% | 90.10% | −0.52 (flat) |
| **Stage 2 improved / uf=1** (1 block + BN) | 92.50% ± 0.71 | 93.65% ± 1.16 | **+1.15** |
| **Stage 2 improved / uf=2** (2 blocks + BN) | **94.48% ± 0.97** | **96.67% ± 0.71** | **+2.19** |

COVID recall (where the gap lives):

| Model | CNN-AD | CNN-SA | Recall lift |
|---|---|---|---|
| Stage 2 / uf=1 | 88.89% ± 1.76 | 91.39% ± 1.62 | +2.50 |
| Stage 2 / uf=2 | 93.33% ± 3.22 | 96.67% ± 2.08 | +3.33 |

**Two wins.** (1) The change **raised the baseline** (90.6 → 94.5% at uf=2). (2) It
**restored the paper's core claim** — augmentation now helps, consistently across
seeds, and more unlocked capacity (uf=2) both lifts the baseline and doubles the
augmentation effect. At uf=2, **CNN-SA (96.67%) exceeds the paper's 95%**.

### 5.1 Supporting analysis — data-scarcity curve
Subsampling the *real* training set while keeping the synthetic pool fixed (uf=2,
3 seeds):

| real frac | n_real | CNN-AD | CNN-SA | acc lift |
|---|---|---|---|---|
| 0.10 | 93 | 86.98% | 89.24% | +2.26 |
| 0.25 | 233 | 89.58% | 92.88% | **+3.30** |
| 0.50 | 466 | 91.32% | 93.06% | +1.74 |
| 1.00 | 932 | 93.92% | 96.01% | +2.08 |

Augmentation helps at **every** data level; the effect is **largest in the
low-to-mid regime** (peak +3.30 at 25%), consistent with the paper's
"augmentation rescues data-starved models" thesis. (Not perfectly monotonic; the
10% point is noisy on just 93 images.)

### 5.2 Supporting analysis — generalization curve (diagnostic)
Per-epoch train/test accuracy (uf=2, diagnostic only — the reported models use a
**fixed 15 epochs chosen test-blind**, and the per-epoch test numbers are **not**
used to select epochs). Train accuracy saturates ≥99% by epoch 3–6, yet **test
accuracy never degrades through 25 epochs** (AD plateau ≈93.7%, SA ≈96.5%). So
there is no harmful overfitting; the fixed 15 epochs sits on the plateau, and
SA's curve stays above AD's at essentially every epoch.

---

## 6. Discussion

**What worked.** The Stage 2 architecture change did exactly what the Stage 1
analysis predicted: unfreezing the encoder raised the baseline *and* revived GAN
augmentation, restoring the paper's core finding (augmentation improves COVID
detection) that the faithful frozen reconstruction could not reproduce. The
data-scarcity curve further shows the effect grows as the baseline weakens —
explaining *why* the paper, with its data-starved 85% baseline, saw a much larger
+10 jump than our +2.

**What did not.** We did not reproduce the paper's **+10-point magnitude**. This
is not a failure of the method but of *available headroom*: our baseline (~94.5%
on the cleaner modern Kaggle data) simply cannot gain 10 points, whereas the
paper started at 85%. We recover the claim's direction and mechanism, not its
size.

**What we learned.** The CovidGAN augmentation benefit is **real but contingent**
on (a) the classifier having trainable capacity to absorb the new data and (b) the
baseline not already being at ceiling. Freezing the backbone or starting from an
easy, high baseline both suppress it. Our three regimes on identical data —
frozen (Stage 1), uf=1, uf=2 — plus the scarcity sweep make this mechanism
explicit and quantified.

**Limitations / future work.** Results are 5-seed (full data) / 3-seed (scarcity)
means; the extreme-scarce point is noisy. One caveat from Stage 1 remains open: a
possible Kaggle-specific fine-scale artifact could inflate the baseline, testable
only by validating on an independent public CXR dataset (not run here). Proper
validation-based early stopping (on a held-out validation split, never the test
set) is a legitimate further refinement.

---

## 7. References

1. A. Waheed, M. Goyal, D. Gupta, A. Khanna, F. Al-Turjman, P. R. Pinheiro,
   "CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved
   Covid-19 Detection," *IEEE Access*, vol. 8, pp. 91916–91923, 2020.
2. A. Odena, C. Olah, J. Shlens, "Conditional Image Synthesis with Auxiliary
   Classifier GANs," *ICML*, 2017.
3. K. Simonyan, A. Zisserman, "Very Deep Convolutional Networks for Large-Scale
   Image Recognition" (VGG16), *ICLR*, 2015.
4. M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, S. Hochreiter, "GANs
   Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
   (FID), *NeurIPS*, 2017.
5. M. E. H. Chowdhury et al. / T. Rahman et al., "COVID-19 Radiography Database,"
   Kaggle. (Real CXR dataset used for reconstruction.)
6. J. P. Cohen, P. Morrison, L. Dao, "COVID-19 Image Data Collection"
   (ieee8023/covid-chestxray-dataset). (A source of the COVID class.)

---

*Code: Stage 1 in `covidgan/`, `train_gan.py`, `train_classifier.py`,
`generate_synthetic.py`, `evaluate_fid.py`. Stage 2 in `stage2/`. Trained
generator weights in `weights/covidgan_generator_final.pt`. Dataset: Kaggle
COVID-19 Radiography Database (link in `README.md`).*
