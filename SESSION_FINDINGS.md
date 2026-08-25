# Session Findings — Stage 2 GAN improvement (AC-GAN fingerprint → projection + noise fix)

*Working log for this session. To be merged into the formal report. Every test below is reproducible from the committed scripts and the commands given. Same-source `data/manifest.csv` (331 COVID / 601 Normal train, 72 / 120 real test = 192) throughout. Device: MPS (Apple M-series).*

---

## 0. Question this session answers

Stage 1 found the GAN's synthetic pool carried **no transferable pathology** — a classifier trained on synthetic alone scored **55% on real** (below the 62.5% majority floor). This session asks **why**, whether it is fixable, and whether a better GAN helps the downstream classifier. Short answer: the failure was a **class "fingerprint"** (a global brightness/contrast difference the generator invented), caused mainly by **near-zero noise + the AC-GAN objective + small-data overfitting, amplified by 2000 epochs**. The improved recipe (noise 1.0 + DiffAugment + spectral norm) and a **projection discriminator** move transfer from **55% → 73–74%**. Downstream, the better pools help **only when the classifier encoder is unfrozen** (+2), not under the paper's frozen head (flat).

---

## Test 1 — Visual inspection of the sample grids (the fingerprint, seen)

Grid layout (all trainers): `eval_labels = arange(16) % 2`, `CLASS_NAMES=["covid","normal"]`, `nrow=4` ⇒ **columns 1 & 3 = COVID, columns 2 & 4 = Normal**; the same 16 fixed noise vectors are reused every epoch.

Files viewed:
- Stage-1 AC-GAN, 2000 ep: `runs/gan/samples/epoch_2000.png`
- Improved AC-GAN, 300 ep: `runs/gan_acgan/samples/epoch_0300.png`
- Improved Projection, 300 ep: `runs/gan_projection/samples/epoch_0300.png`

**Observation.**
- **2000-epoch (noise 0.02):** COVID columns are uniformly **hazy/bright/low-contrast**; Normal columns **dark/sharp**. Near-zero within-class diversity (all 8 COVID tiles ≈ identical; likewise Normal) — mode collapse. The class difference is a **global style**, not localized pathology.
- **300-epoch improved (noise 1.0 + DiffAugment):** within-class diversity restored (tiles differ); the brightness split is much weaker. **Projection** looks the least fingerprinted (classes not separable by any global style), most varied/realistic; **AC-GAN improved** intermediate.

---

## Test 2 — Is the class difference real? (brightness quantification on REAL data)

Real COVID/Normal are **not** originally as different as the 2000-epoch GAN implied.

Reproduce (writes `real_samples.png`, prints class-mean brightness):
```bash
python - <<'PY'
import random, numpy as np
from PIL import Image
from covidgan.data import read_manifest
tr = read_manifest('data/manifest.csv','train')          # (path, int label): 0=covid, 1=normal
cov=[p for p,l in tr if l==0]; nor=[p for p,l in tr if l==1]
load=lambda p: np.asarray(Image.open(p).convert('L').resize((112,112)),dtype=np.float32)
mb=lambda L,n=150: float(np.mean([load(p).mean() for p in random.sample(L,min(n,len(L)))]))
random.seed(7); print('covid',mb(cov),'normal',mb(nor))
PY
```

**Result.**
- Class-mean brightness: **COVID 138.6 vs Normal 129.8 → +8.8 / 255 (~3.5%)** — tiny.
- Per-image means **overlap almost completely**: real COVID 105–168, real Normal 104–144 (darkest COVID < every Normal; brightest Normal > most COVID). **Brightness is not a usable class cue in real X-rays.**
- Real COVID differs by **localized** opacity in specific zones with lung structure preserved — not a whole-image haze.

**Interpretation.** The 2000-epoch GAN **amplified a faint, overlapping +8.8 tendency into a deterministic hazy-vs-dark rule** — manufacturing a separability that does not exist in reality. That is the fingerprint, quantified.

---

## Test 3 — Synthetic-only transfer probe (the decisive test)

Train a classifier on the synthetic pool **alone**, test on **real**. Isolates whether the synthetic *class signal* is transferable pathology or a fingerprint. Script: `stage2/synthetic_only_probe.py` (frozen-VGG16 detector, matching Stage 1's 55% probe). Floor (guess all-Normal) = 62.5%.

```bash
python -m stage2.synthetic_only_probe --synthetic-dir data/synth_acgan      --out-dir runs/probe_acgan
python -m stage2.synthetic_only_probe --synthetic-dir data/synth_projection --out-dir runs/probe_projection
```

| generator | real-test accuracy | vs floor (62.5%) |
|---|---|---|
| Stage-1 AC-GAN, 2000 ep (noise 0.02) | **55%** (COVID recall 0.22) | **below** — pure fingerprint |
| Improved AC-GAN, 300 ep | **72.9%** | above |
| Improved Projection, 300 ep | **74.5%** | above (best) |

Outputs: `runs/probe_acgan/probe_metrics.txt`, `runs/probe_projection/probe_metrics.txt`.

**Interpretation.** The improved recipe moved transfer **55% (below chance) → ~73–74% (clearly above floor)** — synthetic now carries real signal. Projection edges AC-GAN (74.5 vs 72.9), consistent with its less-fingerprinted look, but the 1.6-pt gap (~3 images, single seed) is within noise — **needs multi-seed to claim projection wins.**

---

## Test 4 — FID (generator quality, metric studied in class)

Real = manifest test split (192); synthetic = each pool (3068). *(Small real set biases FID upward — comparison across pools is what matters, not the absolute value.)*

```bash
python evaluate_fid.py --real-manifest data/manifest.csv --real-split test --synthetic-dir data/synthetic
python evaluate_fid.py --real-manifest data/manifest.csv --real-split test --synthetic-dir data/synth_acgan
python evaluate_fid.py --real-manifest data/manifest.csv --real-split test --synthetic-dir data/synth_projection
```

| generator | FID overall | COVID | Normal |
|---|---|---|---|
| Stage-1 AC-GAN, 2000 ep | 272.7 | 302.2 | 290.2 |
| Improved AC-GAN, 300 ep | 225.7 | 248.8 | 251.1 |
| **Improved Projection, 300 ep** | **155.0** | 176.2 | 169.9 |

**Interpretation.** Projection at **300 epochs** beats the Stage-1 AC-GAN at **2000 epochs** by ~43% FID (155 vs 273), and beats improved AC-GAN (226) — the best generator of the three, at a fraction of the epochs. (Absolute FID stays high because the real set is only 192 images ≪ Inception's 2048-dim features — an upward bias; cross-pool *ranking* is the valid read.) Note FID rank (projection ≫ AC-GAN) does **not** translate to a downstream gap (Test 5: 97.4 vs 97.9, tied) — reconfirming FID/quality is not the downstream lever here.

---

## Test 5 — Downstream 2×2: {AC-GAN, Projection} × {frozen, unfrozen}

CNN-SA with each pool, under the paper's **frozen** head (`train_classifier.py`) and the **unfrozen** Track A encoder (`stage2/train_stage2.py --unfreeze-blocks 2`). AD = real-only baseline.

```bash
# frozen (paper design)
python train_classifier.py --mode ad  --out-dir runs/frozen_ad
python train_classifier.py --mode sa  --synthetic-dir data/synth_acgan      --out-dir runs/frozen_sa_acgan
python train_classifier.py --mode sa  --synthetic-dir data/synth_projection --out-dir runs/frozen_sa_proj
# unfrozen (Track A, top 2 VGG blocks)
python -m stage2.train_stage2 --mode ad --unfreeze-blocks 2 --out-dir runs/unfrozen_ad
python -m stage2.train_stage2 --mode sa --unfreeze-blocks 2 --synthetic-dir data/synth_acgan      --out-dir runs/unfrozen_sa_acgan
python -m stage2.train_stage2 --mode sa --unfreeze-blocks 2 --synthetic-dir data/synth_projection --out-dir runs/unfrozen_sa_proj
```

| classifier | AD (real only) | SA + AC-GAN | SA + Projection |
|---|---|---|---|
| **frozen** | 90.62% (rec 0.88) | 90.10% (rec 0.86) | 90.10% (rec 0.82) |
| **unfrozen** | 95.31% (rec 0.92) | **97.92%** (rec 0.97) | 97.40% (rec 0.97) |

*(single seed; the 0.52% frozen "drop" = 1/192 image = noise. Track A 5-seed run confirms the unfrozen augmentation lift at +2.19 ± 0.7: `stage2/results/multiseed_uf2.json`.)*

**Interpretation.** Better GANs help the classifier **only when the encoder is unfrozen** (+2, reproducible). Under the frozen head they stay flat — the frozen-head ceiling, not GAN quality. Projection ≈ AC-GAN downstream (within noise); the projection advantage shows up in *transfer* (Test 3), which the unfrozen "just add data" regime masks.

---

## Test 6 — Does the GAN give ANY improvement? (augmentation lift by data scarcity)

"Any improvement" is already established in the **unfrozen** setup, multi-seed:
- Full data, 5 seeds (`stage2/results/multiseed_uf2.json`): AD 94.48% → SA **96.67% = +2.19 ± 0.8**.
- Scarcity, unfrozen, 3 seeds, old pool (`stage2/results/scarcity_uf2.json`): lift positive at every level — +2.26 (10%), **+3.30 (25%)**, +1.74 (50%), +2.08 (100%). Largest when data is scarce = the paper's regime.

The one flat cell is **frozen head + full data** (Test 5: 90.62→90.10) — no headroom + a ~33K-param head that can't exploit better images.

**This test:** does augmentation help even under the paper's EXACT frozen detector when real data is starved (opening headroom)? Added `--frozen` to `stage2/data_scarcity.py` (uses `covidgan.models.build_classifier`, the paper's frozen-VGG16 detector), run with the **projection** pool:

```bash
python -m stage2.data_scarcity --frozen --synthetic-dir data/synth_projection \
    --fractions 0.1 0.25 0.5 1.0 --seeds 0 1 2 --out-dir runs/scarcity_frozen_proj --epochs 15
```

| real data | CNN-AD (frozen) | CNN-SA (frozen, +projection) | acc lift | recall lift |
|---|---|---|---|---|
| 10% (93) | 83.85% | 77.78% | **−6.08** | −13.89 |
| 25% (233) | 88.37% | 80.21% | **−8.16** | −12.50 |
| 50% (466) | 89.41% | 84.55% | −4.86 | −5.09 |
| 100% (932) | 89.76% | 87.85% | −1.91 | −1.85 |

Output: `runs/scarcity_frozen_proj/summary.json` (3 seeds). **Result: the OPPOSITE of the expectation — under the paper's frozen detector the projection pool *hurts*, and worse as data gets scarcer (−6 to −8 at 10–25%).**

**Interpretation (this is a key finding).** Even the best synthetic pool (FID 155, transfer 74.5%) *degrades* a frozen-head classifier, most when real data is scarce. Two compounding reasons:
1. **The head can't adapt features.** With ~33K params on fixed ImageNet features, the frozen head cannot reconcile real and synthetic — synthetic occupies its own region of VGG feature space (Stage-1 PCA), so the head just fits the synthetic-dominated boundary, which transfers worse to real. Unfreezing lets the encoder reconcile them → the same augmentation flips to **+2…+3.3** (Test 6 unfrozen numbers above).
2. **Dilution.** At 10% real the pool is 3068 synthetic vs 93 real (97% synthetic); the weaker-signal synthetic (74.5% < 90% real) swamps the scarce real set.

**Conclusion.** "Does the GAN give ANY improvement under the paper's exact frozen design?" → **No — it hurts.** The GAN's benefit is real but *conditional on unfreezing the encoder*; the frozen head is the true blocker, not GAN quality. The two Stage-2 changes (better GAN + unfrozen encoder) are **complementary and both required** — neither alone reproduces the paper's augmentation benefit.

*Caveat / open lever:* the negative may be amplified by the extreme synthetic:real ratio under a non-adaptive head. Capping synthetic to ~1× the real count (or augmenting only the minority COVID class) might soften the frozen-head penalty — untested.

---

## Diagnosis — what caused the paper-like model's fingerprint

Ranked by impact:
1. **`noise_std=0.02` (root cause).** z≈0 ⇒ generator ≈ deterministic function of the label ⇒ one prototype per class ⇒ cheapest separable prototypes differ by a **global style** (brightness). Near-zero noise *forces* the fingerprint and kills diversity.
2. **AC-GAN auxiliary-classifier objective.** Rewards any synthetic class separability; a global fingerprint is the path of least resistance. Projection conditioning removes this incentive.
3. **Discriminator overfitting on 403 COVID images.** Memorized real set ⇒ weak G gradient ⇒ collapse. DiffAugment mitigates.
4. **2000 epochs amplified it.** Long training on a collapsed generator sharpens the global split → transfer *below* floor.

## Fixes (state)
- ✅ noise 1.0 — breaks deterministic-prototype collapse (biggest lever; 55→73).
- ✅ projection discriminator (`stage2/gan_projection.py`) — removes fingerprint incentive (74.5, best transfer).
- ✅ DiffAugment + spectral norm — small-data regularization/stability.
- ⏭️ train projection 300→600 (sharpen realism without fingerprint incentive); per-image intensity normalization (forbid the brightness shortcut outright); multi-seed projection-vs-AC-GAN probe for error bars.

## Note on "work like the paper" (+10 downstream)
Not reachable by fixing the GAN alone: the paper's +10 needs its **starved 85% baseline**; our cleaner same-source data starts at 90.6% with no headroom. The *mechanism* still reproduces — augmentation gives +2 in the unfrozen setup, and more on a deliberately starved split (data-scarcity experiment).

---

## Artifacts / file pointers
- New code: `stage2/gan_projection.py`, `stage2/train_gan_manual.py`, `stage2/synthetic_only_probe.py`, `stage2_gan_comparison.ipynb`.
- Trained GANs (300 ep): `runs/gan_acgan/`, `runs/gan_projection/` (checkpoints every 50).
- Pools: `data/synth_acgan/`, `data/synth_projection/` (3068 each); Stage-1 pool `data/synthetic/`.
- Probe results: `runs/probe_acgan/`, `runs/probe_projection/`.
- 2×2 results: `runs/frozen_*`, `runs/unfrozen_*`.
