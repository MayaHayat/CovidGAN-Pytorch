# CovidGAN Reconstruction — Findings & Analysis

*A PyTorch reconstruction of Waheed et al., "CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved Covid-19 Detection", IEEE Access, 2020.*

This document records the substantive findings of the reconstruction and the experiments run to explain an unexpected result. It is intended to feed the "Reconstruction results" and "Discussion" sections of the formal report.

---

## 1. The anomaly / motivating question

The paper's headline result comes from a VGG16-based COVID-19 detector evaluated on real chest X-rays (CXRs) in two configurations:

- **CNN-AD** — trained on **real** COVID/Normal CXRs only ("actual data").
- **CNN-SA** — trained on **real + GAN-synthesized** CXRs ("synthetic augmented").

The paper reports CNN-AD at **85%** accuracy and CNN-SA at **95%**, i.e. a **+10-point** lift attributed to GAN data augmentation.

Our reconstruction's CNN-AD scores **~91%** — **higher** than the paper's baseline of 85%. A faithful reconstruction beating the original baseline is unusual and demands explanation. The bulk of this document investigates **why our baseline is already so strong**, because that fact reframes what a valid test of the paper's augmentation claim would even look like.

---

## 2. Architecture faithfulness (it is NOT a bug)

Before attributing the gap to anything interesting, we verified that the implementation faithfully matches the paper's architecture and training procedure. It does. The gap is therefore **not** an architecture or training-procedure artifact.

| Component | Specification (matches paper Figs. 1–3, Sec. II–III) | Params | Source |
|---|---|---|---|
| **Generator** | Noise branch `z_dim=100` (Fig. 3) → Dense → 7×7×1024; label branch `Embedding(50)` → Dense → 7×7×1; concatenate → four 5×5 stride-2 transpose-conv blocks (7→14→28→56→112), BatchNorm+ReLU, `Tanh` on the last | ~22M | `covidgan/models.py:26` |
| **Discriminator** | Five 3×3 conv blocks (BatchNorm / LeakyReLU(0.2) / Dropout(0.5)), 112→7×7×512, **two heads** (validity + class) = AC-GAN | ~2M | `covidgan/models.py:76` |
| **Classifier** | Frozen ImageNet VGG16 conv base → GlobalAveragePool → Dense(64, ReLU) → Dropout(0.5) → Dense(2) | ~14.7M total, **only ~33K trainable** | `covidgan/models.py:113` |

**Key point — the frozen backbone is faithful, not a shortcut we introduced.** The paper also freezes VGG16 and trains only the custom head. Direct quote (Sec. II-B):

> "The custom layers of the model are trained, without updating the weights of VGG16 layers ... achieved by setting the 'trainable' property on each of the VGG layers to False before training."

Our classifier freezes exactly the VGG16 convolutional base (`covidgan/models.py:120-122`) and trains only the head. Because architecture and training procedure match the paper, the accuracy gap must be a **data effect**, and that is what the rest of this document establishes.

---

## 3. The metric (paper's metric + its meaning)

The paper's headline metric is classification **accuracy** = correct / total on a held-out **real** test set of **192** images (**72 COVID + 120 Normal**). We reproduce this exactly. Alongside it we report precision / recall / F1 / specificity and the confusion matrix (the Table 1 / Fig. 6–7 layout), implemented in `covidgan/metrics.py:11` (`classification_table`). The class-focused metric we add for the classifier is **per-class recall / F1**; for the GAN we add **FID** in a separate script.

Paper CNN-AD: **164/192 = 85.42%**, with **COVID recall 0.69** — it missed **22 of 72** COVID cases.

Our CNN-AD (same-source split): **175/192 = 91.15%**, with **COVID recall 0.89** — it missed only **8 of 72** COVID cases. Normal recall is essentially saturated in both. **The entire accuracy gap lives in COVID recall.**

| | Accuracy | COVID recall | COVID missed (of 72) |
|---|---|---|---|
| **Paper CNN-AD** | 85.42% (164/192) | 0.69 | 22 |
| **Our CNN-AD** | **91.15% (175/192)** | **0.89** | **8** |

The question "why do we beat the paper?" is therefore precisely the question "**why do we catch far more COVID cases?**"

---

## 4. Hypothesis 1 — cross-source shortcut (TESTED, REJECTED as main cause)

**The concern.** The paper (and our default multi-source pipeline) draws COVID images from the IEEE `covid-chestxray-dataset` and Normal images from a **different** dataset. When the two classes come from different sources, a classifier can separate them on **non-pathological** cues — resolution, borders, embedded text, brightness / processing curves — a "shortcut" that inflates accuracy without learning lung pathology. This is a well-known failure mode of early COVID-CXR classifiers, and it was our leading suspect for the inflated baseline. The default multi-source collection is documented as vulnerable to exactly this (`prepare_dataset.py:18-23`).

**The test — a same-source A/B.** We added a single-source collection path, `load_same_source` (`covidgan/data.py:80`), wired into `prepare_dataset.py` via `--same-source-root` (`prepare_dataset.py:52`). It draws **both** classes from **one** dataset's per-class subfolders — the Kaggle COVID-19 Radiography Database's `COVID/` and `Normal/` directories — so both classes pass through **one acquisition + post-processing pipeline**, eliminating the cross-source shortcut. We subsampled to the paper's scale (**403 COVID + 721 Normal**, `--max-covid/--max-normal`) and kept the paper's test counts (**72 / 120**).

**Result.** Same-source CNN-AD **still scored 91.15%**. Removing the cross-source path did **not** collapse accuracy toward 85%.

**Verdict.** The cross-source shortcut is **not** the main driver of the gap. (This is a genuinely reassuring result: it means the strong baseline is not simply the classic "different-scanner" artifact.)

---

## 5. Hypothesis 2 — coarse-resolution / global shortcut (TESTED via Experiment 9a)

**The concern.** Even within one source, a classifier can cheat on **coarse / global** properties (overall brightness, framing, gross silhouette) rather than reading pathology.

**Method.** Downsample every image to N×N (area interpolation), then upsample back to 112 (bilinear). This destroys fine anatomical detail while preserving coarse / global structure. Retrain CNN-AD at each N. The majority-class floor (guess all-Normal) is **120/192 = 62.5%**.

| Input detail | Accuracy | COVID recall |
|---|---|---|
| 112 (full) | 89.58% | 0.83 |
| 32×32 | 83.85% | 0.64 |
| 16×16 | 82.29% | 0.61 |
| 8×8 | 77.08% | 0.51 |
| 4×4 | 73.96% | 0.51 |

**Interpretation.** Accuracy **decays steadily** (−16 points from full → 4×4), and COVID recall roughly **halves** (0.83 → 0.51). A *pure coarse / global shortcut* would keep accuracy near 89% even at 8×8 — brightness and framing survive aggressive downsampling — but instead accuracy collapses, and it collapses **specifically on COVID** (Normal stays easy at any resolution). This is the signature of a **detail-dependent** model reading real fine / mid-scale pathology, not a coarse shortcut.

**But** 4×4 (16 pixels, no anatomy whatsoever) still beats the 62.5% floor by ~11 points, so a **small coarse / global residual** exists.

**Verdict.** Mostly detail-dependent, with a minor coarse residual.

---

## 6. Combined explanation of the 91% vs 85% gap

Pulling the two tests together:

- **Not a cross-source artifact** — same-source also scores 91.15% (§4).
- **Not a coarse-resolution shortcut** — Experiment 9a shows detail-dependence, with only a small coarse residual (§5).

The most consistent explanation is that the **modern Kaggle COVID-19 Radiography Database is cleaner, larger, and more separable** than the paper's 2020 hand-merged small dataset (**403 COVID + 721 Normal** assembled from three public sources and de-duplicated). Our model performs **legitimate, detail-based** classification on an **easier** dataset, and the improvement is concentrated exactly where §3 located it — **COVID recall**.

---

## 7. Open question / caveat (what is NOT yet proven)

Experiment 9a rules out **coarse** shortcuts but **not** a **fine-scale, dataset-specific artifact** — e.g. a rescaling-kernel signature, a JPEG/compression fingerprint, or a faint embedded text / watermark. Such an artifact lives in fine detail, would **also vanish at 8×8**, and would therefore **masquerade as detail-dependence** in the 9a table. In other words, 9a cannot distinguish "reading transferable COVID pathology" from "reading a Kaggle-specific fine fingerprint."

The decisive test is **Experiment 9b — cross-dataset validation**: evaluate the CNN-AD on an **independent** public dataset (both classes, non-overlapping with Kaggle/IEEE). If accuracy holds, the signal is transferable pathology; if it collapses, the baseline was riding a Kaggle-specific fingerprint. **9b has not yet been run.**

**Important constraint.** The IEEE `covid-chestxray-dataset` **cannot** serve as the independent set: per the Kaggle README it is one of the **sources** of the Kaggle COVID class (so the two overlap), and it has **no Normal class** at all. A truly independent third dataset is required.

---

## 8. The GAN augmentation arm (CNN-SA) — trained to the paper's 2000 epochs

The GAN has now been trained for the paper's full **2000 epochs** (batch 64, lr 2e-4, Adam β₁=0.5, one-sided label smoothing), checkpointing every 100 epochs. On the M-series GPU (MPS) this took ~18 hours at ~55 min per 100 epochs; the run was made interruption-safe with checkpoints that store optimizer state and a `--resume` flag (`train_gan.py`). The trained generator (epoch 2000) is committed at `weights/covidgan_generator_final.pt`, and the synthetic pool was regenerated from it (**1,669 COVID + 1,399 Normal = 3,068** images, matching the paper).

### 8.1 The generator did learn — FID dropped ~2×

FID (Fréchet Inception Distance; lower = closer to the real image distribution) between the real test CXRs and the synthetic pool, before (25-epoch smoke GAN) and after (2000-epoch GAN):

| Set | 25-epoch (noise) | **2000-epoch** | Change |
|---|---|---|---|
| overall | 504.4 | **272.7** | −46% |
| COVID | 487.2 | 302.2 | −38% |
| Normal | 538.0 | 290.2 | −46% |

FID roughly **halved** — the 2000-epoch generator produces meaningfully more CXR-like images than the near-noise smoke GAN. In absolute terms ~273 is still high (a good FID is single / low-double digits): partly the upward bias of a small real set (72–120 images ≪ InceptionV3's 2048-dim features, which `evaluate_fid.py` warns about), and partly genuine — 2000 epochs on 403 COVID images at 112×112 yields *plausible* CXRs, not photorealistic ones.

### 8.2 …yet CNN-SA does not beat CNN-AD — the result is flat

Retraining the detector on real + the **2000-epoch** synthetic pool:

| Model | Accuracy | COVID recall | Normal recall |
|---|---|---|---|
| **CNN-AD** (real only) | 90.62% (174/192) | 0.889 (64/72) | 0.917 (110/120) |
| **CNN-SA** (+ 2000-epoch synthetic) | 90.10% (173/192) | 0.861 (62/72) | 0.925 (111/120) |

(The paired CNN-AD baseline here is 90.62%, within run-to-run variance of the 91.15% in §3 — a ±1-image wobble on 192 test images.) CNN-SA is **flat** — actually one image lower, trading 2 COVID catches for 1 Normal. That 0.52-point difference is **within run-to-run noise** (each image = 0.52%): *no improvement, and no reliable degradation* — not a real drop.

Head-to-head with the paper:

| | Paper | Our reconstruction |
|---|---|---|
| CNN-AD | 85% | 90.6% |
| CNN-SA | 95% (**+10**) | 90.1% (**flat**) |

### 8.3 Why the performance did not change

Four compounding reasons, most important first:

1. **We are already near the ceiling; the paper was not.** The paper's premise is *small, hard, data-starved* — its CNN-AD sat at **85%** with ample headroom for augmentation to fill. Our baseline is already **~90.6%** on the cleaner, more separable modern Kaggle data (§3–§6). A +10-point jump simply isn't available from a 90.6% start.
2. **The bottleneck here is not data quantity/diversity.** Augmentation helps a classifier that is starved for examples. Our frozen-VGG features already separate the classes cleanly at ~90%, so extra synthetic samples mostly reinforce a boundary that is already well-placed rather than filling a deficit.
3. **The frozen backbone caps augmentation's leverage.** Only ~33K head parameters train (the VGG16 base is frozen, per the paper), so synthetic images can only nudge a small linear boundary — they cannot reshape learned features.
4. **Decisive evidence that image *quality* is not the lever.** FID **halved** (noise → plausible) between the smoke GAN and the 2000-epoch GAN, yet CNN-SA landed at **90.1%** — statistically identical to *both* CNN-AD **and** the earlier noise-GAN CNN-SA (91.15% / COVID recall 0.86). If image quality drove the outcome, a 2× FID improvement should have produced *some* downstream lift; it produced none. This isolates the cause as a **headroom / frozen-head ceiling**, **not** poor image quality.

### 8.4 Supporting diagnostics (from the smoke-GAN phase)

These earlier probes explain the *mechanism* and still hold:

- **High CNN-SA *train* accuracy (~99%) is not evidence of good images.** Synthetic images are **77%** of the training set (932 real + 3068 synthetic). An AC-GAN imprints a **class-conditioned** signature via the label embedding, so synthetic COVID vs. Normal are trivially separable — inflating *train* accuracy regardless of realism.
- **Synthetic-only probe.** A classifier trained on the (smoke-GAN) synthetic pool alone reached **100%** synthetic-train accuracy but only **55.21%** on the real test set (COVID recall **0.22**) — below the 62.5% majority floor — confirming the synthetic class signal carries essentially zero transferable pathology.
- **Why CNN-SA never collapses**, even on bad data: the frozen backbone limits corruption to ~33K params; the synthetic images occupy their own region of VGG16 feature space (PCA, `covidgan/metrics.py:95`); so they barely perturb the real decision boundary.

---

## 9. Reproducibility / performance note

The original scripts selected the compute device with only `torch.cuda.is_available()`, so on Apple-silicon Macs they **silently ran on CPU** (the M-series GPU sat idle), and the dataset was **re-decoded and re-resized every epoch**. We added:

- **`pick_device`** (`covidgan/models.py:135`) — prefers **cuda > mps > cpu**, so M-series Macs use the Metal GPU.
- **An in-RAM dataset cache** (`CXRDataset(cache=True)`, `covidgan/data.py:189`; default-on in both trainers, e.g. `train_classifier.py:154`) — every image is decoded and resized **once** at construction, so each epoch reads pre-made tensors.

**Effect.** CNN-AD training dropped from **~19 min to ~1 min** on an M4. Critically, a Colab **T4 was previously no faster than the laptop** because the GPU was starved by CPU image decoding; caching removes that bottleneck, making the full **2000-epoch GAN run practical on a free Colab GPU**.

---

## Status / next steps

1. ✅ **GAN trained to 2000 epochs** (§8) — generator committed at `weights/covidgan_generator_final.pt`.
2. ✅ **Synthetic pool regenerated + FID computed** — FID 504 → 273 (§8.1), confirming the generator genuinely learned.
3. ✅ **CNN-SA rerun** on the real-quality synthetic data (§8.2) — **flat, 90.6 → 90.1**, explained by the ceiling / frozen-head effect (§8.3).
4. **(Recommended) Multi-seed comparison** — 5× CNN-AD vs. 5× CNN-SA to report mean ± spread, so "flat" is backed statistically rather than by a single run.
5. **(Optional but decisive) Experiment 9b** — cross-dataset validation on an **independent** public dataset (§7), to close the "transferable pathology vs. Kaggle-specific fingerprint" question.
6. **Stage 2 (improvement)** — deliberately create headroom so augmentation *can* help: e.g. unfreeze some VGG16 layers, or use a harder split, then re-test the 85→95-style claim.
