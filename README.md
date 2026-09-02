# Vineyard Scene Reconstruction

> Repurposing existing precision-viticulture UAV/ground-robot footage — none of it collected with 3D reconstruction in mind — into 3D Gaussian Splatting scenes of real grapevine rows, for the DRAGONS project's downstream goal of RL-based UAV/robot grape-sampling simulation. Stack: **pycolmap** (SfM) · **gsplat** / **Nerfstudio splatfacto** (3DGS) · **uv** · **SLURM**.

## 🍇 What this is

Public vineyard datasets (UAV video, multispectral UAV imagery, ground-robot RGB-D) were collected for detection, tracking, or disease mapping — not for radiance-field reconstruction. This project asks whether they can be reconstructed anyway, catalogues what breaks when they can't, and fixes what's fixable with low-effort preprocessing rather than new captures.

Three things live here:
1. A **dataset survey** ([`docs/datasets/`](docs/datasets/)) scoring public vineyard datasets on multi-view structure, real-scene content, season/grape-visibility, and reconstruction suitability.
2. A **standardized reconstruction pipeline** ([`src/reconstruction/`](src/reconstruction/)) — video/imagery → frames → COLMAP SfM → 3DGS (gsplat or Nerfstudio splatfacto), with a fixed protocol so cross-dataset comparisons reflect dataset content, not pipeline tuning.
3. A **failure catalogue with fixes** — per-dataset breakdown of what reconstructs cleanly, what doesn't, and why: a lens-flare/veiling-glare + lens-ghost artefact on UAV footage ([`docs/flare_fix_plan.md`](docs/flare_fix_plan.md), [`docs/flare_fix_research.md`](docs/flare_fix_research.md)), band misalignment on a 5-lens multispectral camera ([`docs/datasets/multispectral_botrytis.md`](docs/datasets/multispectral_botrytis.md)), and narrow-FOV/low-overlap failure on ground-robot sequences (`paper/sections/figure2_failure.tex`).

Write-up in progress at [`paper/`](paper/) (standalone LaTeX, not yet merged into the main DRAGONS paper).

## 🛠️ Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Sync the reconstruction pipeline's dependencies:
   ```sh
   uv sync --extra recon        # pycolmap, gsplat, opencv, lpips/ssim metrics
   uv sync --extra recon-ns     # + Nerfstudio, for the splatfacto training path
   uv run pre-commit install
   ```
   Both extras can be installed together (`uv sync --extra recon --extra recon-ns`). `recon-ns` is only needed if you're training with `ns-train splatfacto` rather than the custom `train_gs.py` loop.
3. On a SLURM cluster, `module load CUDA/13.0.0` (or whatever CUDA module your cluster exposes) before any GPU step — `gsplat`/`nerfstudio` silently fall back to CPU-disabled builds otherwise.

## 🚀 Quickstart: reconstruct a scene

The fastest way to see the pipeline work end-to-end is the smoke test — one small UAV clip, short training budget, no dataset download required beyond the clip itself:

```sh
sbatch slurm/smoke_test_reconstruction.sh
```

That runs, in order, the same three stages any scene goes through:

```sh
# 1. Sample frames from source video at a fixed fps/resolution (configs/reconstruction.yaml)
uv run --extra recon python src/reconstruction/extract_frames.py <video.mp4> <frames_dir>

# 2. COLMAP structure-from-motion (feature extraction, exhaustive matching, incremental mapping)
uv run --extra recon python src/reconstruction/sfm.py <frames_dir> <colmap_dir>

# 3a. Train with the custom gsplat loop (holds out every Nth view, reports PSNR/SSIM)
uv run --extra recon python src/reconstruction/train_gs.py \
    "$(cat <colmap_dir>/best_sparse_dir.txt)" <frames_dir> <output_dir> --iters 30000

# 3b. ...or convert to a Nerfstudio dataset and train splatfacto instead
uv run --extra recon python src/reconstruction/to_nerfstudio.py <sparse_dir> <frames_dir> <ns_data_dir>
uv run --extra recon-ns ns-train splatfacto --data <ns_data_dir> \
    --max-num-iterations 30000 nerfstudio-data --eval-mode interval --eval-interval 8
```

Sampling rate, resize, frame cap, COLMAP camera model, and training/holdout settings are all pulled from [`configs/reconstruction.yaml`](configs/reconstruction.yaml) by default, so a scene only deviates from the standard protocol when a dataset-specific fix requires it (documented per-dataset when that happens).

To go beyond the smoke test, pull one of the surveyed datasets — see [`docs/datasets/README.md`](docs/datasets/README.md) for the full evaluation table, access links, and `slurm/download_datasets.sh` for the bulk-download script.

## 🔧 Known failure modes and fixes

Not every dataset reconstructs cleanly out of the box. Two preprocessing pipelines exist for the failure modes found so far — both validated at the pixel level (before/after stills in [`paper/figures/`](paper/figures/)), independent of whether they're wired into the paper's numbers yet:

- **Lens flare / veiling glare + lens ghost** (MOTS UAV footage, backlit sun) — `src/reconstruction/deflare_dark_channel.py` (dark-channel-prior deglare) + `src/reconstruction/generate_ghost_masks.py` (ghost region mask). Full investigation and arm-by-arm numbers: [`docs/flare_fix_research.md`](docs/flare_fix_research.md).
- **Multispectral band misalignment** (Micasense RedEdge-MX, 5 physically separate lenses, no embedded calibration) — `src/reconstruction/extract_botrytis_rgb.py` (ORB-matched per-band alignment + joint color stretch). Details: [`docs/datasets/multispectral_botrytis.md`](docs/datasets/multispectral_botrytis.md).

Current cross-dataset outcomes (held-out PSNR/SSIM under the standardized protocol) are tracked in [`paper/sections/outcomes_table.tex`](paper/sections/outcomes_table.tex) as results land.

## 📁 Repo map

```
docs/datasets/        Dataset survey: suitability, season/grape-visibility, access links
docs/flare_fix_*.md   Lens-flare/ghost investigation: plan + running research log
docs/harness.md       Reference for the generic Hydra/W&B experiment harness (below)
configs/reconstruction.yaml   Standardized frame-sampling / COLMAP / training protocol
src/reconstruction/   The actual pipeline: extract → SfM → train/eval → per-dataset fixes
slurm/                One SLURM script per pipeline stage / dataset / fix arm
paper/                Standalone LaTeX write-up (figures, outcomes table, discussion)
```

`src/` also carries a generic Hydra + Weights & Biases experiment harness (`main.py`, `model/`, `dataset/`, `metric/`, `logger/`, `configs/{model,dataset,metric,logger,hydra}/`) inherited from the project template — not used by the reconstruction pipeline above, which runs as standalone scripts. It's kept around for downstream RL-policy training on the reconstructed scenes (the project's eventual goal); see [`docs/harness.md`](docs/harness.md) if you're building on that side instead.

## 📚 Citation

Citation details will be added once the write-up (`paper/`) is finalized and merged into the main DRAGONS paper.
