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

## 8. The GAN augmentation arm (CNN-SA) and why it can't yet be evaluated

The paper trains CovidGAN for **2000 epochs** (~5h on an RTX 2060; this is the default in `train_gan.py:114`). Our GAN so far has run only **~25 epochs** as a smoke test, so its synthetic images are **near-noise**. Every CNN-SA finding below must be read in that light: the augmentation arm cannot yet be validly evaluated, and the numbers instead serve as diagnostics of *how a frozen-backbone classifier reacts to bad synthetic data*.

**CNN-SA is flat, and slightly harmful on COVID.** Trained on real + this near-noise synthetic pool, CNN-SA scores **91.15%** — identical accuracy to CNN-AD — but it **traded 2 COVID detections for 2 Normals** (COVID recall 0.86 vs. CNN-AD's 0.89). No lift; slight COVID harm.

| Confusion matrix (rows = true, cols = predicted) | predicted COVID | predicted Normal |
|---|---|---|
| **CNN-AD** — true COVID | 64 | 8 |
| **CNN-AD** — true Normal | 9 | 111 |
| **CNN-SA** — true COVID | 62 | 10 |
| **CNN-SA** — true Normal | 7 | 113 |

Both total **175/192 = 91.15%**.

**Why CNN-SA's *train* accuracy looked high (~99%) despite bad images.** Synthetic images made up **77%** of the training set (**932 real + 3068 synthetic**). An AC-GAN produces **class-conditioned** structured noise — the label embedding imprints a systematic, per-class difference into even a poorly-trained generator's output — so synthetic COVID vs. synthetic Normal are **trivially separable**. High train accuracy is an artifact of that trivial separability and is **not evidence the images are good**.

**Decisive probe — synthetic-only classifier.** We trained a classifier on the **synthetic pool only** (3068 images). It reached **100%** synthetic-train accuracy but only **55.21%** on the **real** test set (COVID recall **0.22**) — **below** the 62.5% majority floor. This proves the synthetic class signal is a **GAN artifact with essentially zero transferable pathology**.

**Why CNN-SA doesn't collapse despite training on bad data.** Three reasons compound:

1. **Frozen backbone** — only ~33K head params can be affected at all, so the synthetic noise has very little capacity to corrupt.
2. **Feature-space separation** — the synthetic-noise images occupy their own region of VGG16 feature space (visible in the PCA plot from `plot_pca`, `covidgan/metrics.py:95`), so they barely perturb the real decision boundary.
3. **Short training** — 25 epochs still let the head fit the real images.

**Consequence.** To validly test the paper's 85→95 augmentation claim, the GAN **must** be trained to ~2000 epochs — until samples resemble real CXRs, judged by the new **FID** metric. And because our baseline is **already 91%** (vs. the paper's 85%), there is **little headroom** for a +10-point jump. The meaningful thing to watch is therefore **whether CNN-SA improves COVID recall**, not absolute accuracy.

---

## 9. Reproducibility / performance note

The original scripts selected the compute device with only `torch.cuda.is_available()`, so on Apple-silicon Macs they **silently ran on CPU** (the M-series GPU sat idle), and the dataset was **re-decoded and re-resized every epoch**. We added:

- **`pick_device`** (`covidgan/models.py:135`) — prefers **cuda > mps > cpu**, so M-series Macs use the Metal GPU.
- **An in-RAM dataset cache** (`CXRDataset(cache=True)`, `covidgan/data.py:189`; default-on in both trainers, e.g. `train_classifier.py:154`) — every image is decoded and resized **once** at construction, so each epoch reads pre-made tensors.

**Effect.** CNN-AD training dropped from **~19 min to ~1 min** on an M4. Critically, a Colab **T4 was previously no faster than the laptop** because the GPU was starved by CPU image decoding; caching removes that bottleneck, making the full **2000-epoch GAN run practical on a free Colab GPU**.

---

## Status / next steps

1. **Train the GAN to ~2000 epochs** on a Colab GPU (now practical thanks to §9), until samples look like real CXRs.
2. **Regenerate the synthetic pool** and **recompute FID** to confirm sample quality objectively.
3. **Rerun CNN-SA** on the real-quality synthetic data and compare against the paper's 85→95 claim — watching **COVID recall**, since absolute-accuracy headroom is small.
4. **(Optional but decisive) Run Experiment 9b** — cross-dataset validation on an **independent** public dataset — to close the remaining "transferable pathology vs. Kaggle-specific fingerprint" question from §7.
