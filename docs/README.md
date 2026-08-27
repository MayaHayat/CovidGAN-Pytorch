# Documentation index

Map of everything under `docs/`. Code lives at the repo root (`covidgan/`,
`stage2/`, the entrypoint scripts) and in `notebooks/`.

## `reports/` — the deliverables (canonical)

The current Part-2 joint report (CovidGAN + DDSM), in four forms plus the
self-contained Overleaf bundles.

| File | What it is |
|------|------------|
| `REPORT.md` | **Primary report** — full unified joint report. Start here. |
| `report_short.md` | Condensed version of the same report. |
| `report.tex` | LaTeX of the full report. |
| `report_short.tex` | LaTeX of the condensed report. |
| `overleaf/joint/` | Self-contained Overleaf bundle for the full report (`report.tex` + figures). |
| `overleaf/short/` | Self-contained Overleaf bundle for the condensed report. |

The `.md` reports link figures from `../../stage2/results/figures/`. The `.tex`
files use flat figure names and are meant to compile from their `overleaf/`
bundle, where the figures sit alongside.

## `findings/` — working research logs

Raw investigation notes that fed the reports. Kept for provenance and reuse.

| File | Covers |
|------|--------|
| `FINDINGS.md` | Stage 1 reconstruction: the anomaly (baseline 90.6% vs paper's 85%) and the experiments explaining it. |
| `SESSION_FINDINGS.md` | Stage 2 GAN-architecture line: the AC-GAN "fingerprint" fix (noise 0.02→1.0, projection discriminator, DiffAugment). |

See also `stage2/STAGE2_FINDINGS.md`, which stays inside the `stage2/` package as its self-contained results doc.

## `archive/` — superseded drafts

Earlier report drafts, replaced by `reports/REPORT.md`. Kept for history; not maintained.

| File | Superseded by |
|------|---------------|
| `JOINT_REPORT.md` | `reports/REPORT.md` |
| `UNIFIED_REPORT.md` | `reports/REPORT.md` |

## `assignment/` — the course brief

`PinalProjectPart2.md` (extracted text) and `PinalProjectPart2.pdf` (original).
The PDF is gitignored (not part of the submission repo).

## `references/` — background sources

- `2103.05094v1.pdf` — DiffAugment (arXiv:2103.05094), used in the Stage 2 GAN recipe.
- `GAN_Medical_Imaging_Paper_Survey.md` — paper-selection survey.

Both are gitignored as personal/reference material.
