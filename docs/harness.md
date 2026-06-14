# Experiment Harness Reference

This document describes the full structure of the experiment harness and serves as a pre-flight checklist before writing any research code in this codebase.

---

## Codebase Map

```
<project>/
├── configs/                        # Hydra config tree (compose-time)
│   ├── config.yaml                 # Root config — sets defaults + exp.run_func
│   ├── experiment/default.yaml     # Experiment-level overrides (seed, device)
│   ├── model/default.yaml          # _target_ for model class + constructor kwargs
│   ├── dataset/default.yaml        # _target_ for dataset class + DataLoader kwargs
│   ├── metric/default.yaml         # Map of name → _target_ for each metric class
│   ├── logger/wandb/default.yaml   # WandBLogger kwargs (project, entity, …)
│   └── hydra/default.yaml          # Hydra run.dir path template (MUST be set)
├── src/
│   ├── main.py                     # Entry point — hydra.main + call(exp.run_func)
│   ├── experiment/run.py           # Main loop (setup → components → loop → metrics)
│   ├── model/base.py               # BaseModel (ABC + nn.Module)
│   ├── dataset/base.py             # BaseDataset (ABC + Dataset)
│   ├── metric/base.py              # BaseMetric (ABC, update + compute_and_log)
│   ├── logger/
│   │   ├── base.py                 # BaseLogger (ABC, log(dict))
│   │   └── wandb.py                # WandBLogger — metrics/ → W&B, metadata/ → CSV
│   └── utils/
│       ├── __init__.py             # re-exports hydra, training, wandb submodules
│       ├── hydra.py                # preprocess_config(): sets log_dir + symlink
│       ├── training.py             # setup_device, set_seed, save/load_checkpoint
│       └── wandb.py                # setup_logger(): instantiates logger from config
├── slurm/
│   ├── SLURM_script.sh             # Generic sbatch wrapper (generates .slurm file)
│   ├── run_experiment.sh           # Single-experiment launcher (called by sbatch)
│   └── resource_logger.py          # Background RAM/VRAM CSV logger + PNG plot
├── notebooks/                      # Scratch notebooks (not tracked)
├── pyproject.toml                  # uv/setuptools project; ruff config
├── .python-version                 # 3.11 (pinned for uv)
└── .pre-commit-config.yaml         # ruff-check (--fix) + ruff-format on every commit
```

---

## How a Run Flows

```
uv run python src/main.py [hydra overrides]
        │
        ▼
main()  ──hydra.main──►  compose configs from configs/config.yaml
                          resolve env vars MODEL, DATASET
        │
        ▼
call(config.exp.run_func)  →  experiment.run.run(config)
        │
        ├─ utils.hydra.preprocess_config(config)
        │     sets config.exp.log_dir = Hydra output dir
        │     creates logs/<date>/<time> symlink in CWD
        │
        ├─ utils.wandb.setup_logger(config)
        │     instantiate(config.logger)(config=..., dir=..., group=date, name=time)
        │
        ├─ utils.training.setup_device(config)   # cuda > mps > cpu
        ├─ utils.training.set_seed(config.exp.seed)
        │
        ├─ instantiate(config.model).to(device)
        ├─ instantiate(config.dataset)  →  DataLoader(dataset, **config.dataset.dataloader)
        ├─ {name: instantiate(cfg) for name, cfg in config.metric.items()}
        │
        ├─ [MAIN LOOP]  ← fill in here
        │
        └─ for name, metric in metrics.items():
               result = metric.compute_and_log()
               logger.log({f"metrics/{name}": result})
```

---

## Pre-Flight Checklist

Work through this list in order before writing any research code.

### 1. One-time Project Setup
- [ ] Rename `name = "project"` in [pyproject.toml](../pyproject.toml) to the actual project name.
- [ ] Set `run.dir` in [configs/hydra/default.yaml](../configs/hydra/default.yaml) to the real storage path (e.g. `/scratch/<user>/<project>/${now:%Y-%m-%d}/${now:%H-%M-%S}`).
- [ ] Set W&B `project` (and `entity` if needed) in [configs/logger/wandb/default.yaml](../configs/logger/wandb/default.yaml).
- [ ] Run `uv sync && uv run pre-commit install` once per machine.
- [ ] Fill in `#SBATCH --account=FILL_MANUALLY` in the generated SLURM scripts (see [slurm/SLURM_script.sh](../slurm/SLURM_script.sh) line 160).

### 2. Config Hygiene
- [ ] `config.yaml` selects model/dataset via env vars `MODEL` and `DATASET` (`${oc.env:MODEL,default}`). Every new variant YAML placed in `configs/model/` or `configs/dataset/` is automatically selectable — no Python change needed.
- [ ] `configs/logger/wandb/default.yaml` must be present at Hydra compose time or the run will fail with a missing config group error.
- [ ] `config.exp.log_dir` is `null` at compose time and set at runtime by `preprocess_config()` — never set it in YAML.
- [ ] `config.exp.device` is `null` at compose time — auto-selects cuda > mps > cpu. Override with `exp.device=cpu` on the CLI when needed.

### 3. Implementing a Model
- [ ] Subclass `BaseModel` ([src/model/base.py](../src/model/base.py)) in a new file under `src/model/`.
- [ ] Implement `forward(*args, **kwargs)`.
- [ ] Create `configs/model/<name>.yaml` with `_target_: model.<module>.<ClassName>` plus any constructor kwargs.
- [ ] Select at runtime: `MODEL=<name> uv run python src/main.py` or `model=<name>` CLI override.

### 4. Implementing a Dataset
- [ ] Subclass `BaseDataset` ([src/dataset/base.py](../src/dataset/base.py)) in a new file under `src/dataset/`.
- [ ] Implement `__len__()` and `__getitem__(idx)`.
- [ ] Create `configs/dataset/<name>.yaml` with `_target_` plus constructor kwargs AND the `dataloader:` block (batch_size, num_workers, shuffle, pin_memory).
- [ ] The DataLoader is built in `run.py` directly from `config.dataset.dataloader` — keep that block in every dataset config.
- [ ] Select at runtime: `DATASET=<name> uv run python src/main.py`.

### 5. Implementing Metrics
- [ ] Subclass `BaseMetric` ([src/metric/base.py](../src/metric/base.py)) in a new file under `src/metric/`.
- [ ] Implement `update(*args, **kwargs)` — called inside the loop to accumulate state.
- [ ] Implement `compute_and_log() -> dict` — called once at the end; return `{metric_name: scalar}`.
- [ ] Register in `configs/metric/default.yaml`:
  ```yaml
  my_metric:
    _target_: metric.my_module.MyMetric
    # constructor kwargs here
  ```
- [ ] `run.py` iterates all metrics and logs each result as `metrics/<name>`.

### 6. Filling in the Main Loop
- [ ] `src/experiment/run.py` has a placeholder `for batch in dataloader: pass`. Replace this section.
- [ ] Call `metric.update(batch, output)` inside the loop for any metrics that need per-batch state.
- [ ] Log scalars in-loop with `logger.log({"metrics/loss": value})`.
- [ ] Log tensors/arrays with `logger.log({"metadata/preds": tensor})` → saved as `preds.csv` in the run dir.
- [ ] Do NOT call `metric.compute_and_log()` inside the loop — it is called once at the end.

### 7. Logging Contract
The `WandBLogger.log(dict)` dispatcher routes by key prefix:

| Key prefix | Destination | Notes |
|---|---|---|
| `metrics/<name>` | `wandb.log` scalar | Stripped to `<name>` before logging |
| `metadata/<name>` | CSV file in run dir | Appended row-by-row; tensors forced to numpy |

- Keys that match neither prefix are silently dropped.
- The logger respects `exclude_metrics` and `exclude_metadata` lists set in the config.
- `wandb.init` receives the full resolved config dict as its `config` argument — every hyperparameter is automatically captured.

### 8. Reproducibility
- `set_seed(config.exp.seed)` seeds Python `random`, NumPy, and PyTorch (including CUDA).
- Hydra writes a `.hydra/config.yaml` snapshot inside each run's output dir — exact config is always recoverable.
- `save_checkpoint` / `load_checkpoint` in `utils.training` handle model + optimizer state.
- The `logs/<date>/<time>` symlink in the CWD makes it easy to locate the latest run logs without knowing the storage path.

### 9. SLURM
- Entry point: `bash slurm/SLURM_script.sh --script slurm/run_experiment.sh --params "..." --time HH:MM:SS --mem XGB --gpu N`
- Hydra overrides are passed as the `--params` string, forwarded verbatim to the Python command.
- `MODEL` and `DATASET` env vars are forwarded inside `run_experiment.sh`.
- Log files land in `logs/<script_name>/<job_id>/`.
- Add `--track-mem` to spawn `resource_logger.py` as a background process — produces `resource_usage.csv` and `resource_usage.png` in the job log dir.
- `--array 1-N` for array jobs; `--dependency JOBID` for sequential chaining.
- `--print-only` dry-runs without submitting — use this to verify the generated `.slurm` script first.

### 10. Code Quality
- **Linter/formatter**: `ruff` (line length 100). Runs automatically on every commit via pre-commit.
- Enabled rule sets: `E`, `W`, `F`, `N` (pep8-naming), `I` (isort), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify), `NPY`, `RUF`.
- Ignored: `N803`, `N806`, `N812` — uppercase variable/argument names are allowed (common in ML for matrices).
- Python ≥ 3.11 is required and pinned in `.python-version`.

---

## Common Mistakes to Avoid

- **Adding a new component without a YAML**: Hydra instantiates everything via `_target_`. A Python class with no matching YAML is invisible to experiments.
- **Putting `log_dir` in YAML**: It is set at runtime. Hard-coding it in YAML will be overwritten and can cause path confusion.
- **Calling `metric.compute_and_log()` in the loop**: Metrics accumulate state via `update()`; `compute_and_log()` is the final aggregation step.
- **Logging without a namespace prefix**: Keys that don't start with `metrics/` or `metadata/` are silently dropped by `WandBLogger`.
- **Forgetting to set `run.dir`**: Hydra will write outputs to the current directory rather than the storage location, filling up home/scratch quotas.
- **Not setting W&B account/project before first run**: `wandb.init` will prompt interactively on HPC, blocking the job indefinitely.
- **Using `--track-mem` without `psutil` / `nvidia-ml-py`**: The resource logger loads these via `uv run --with`, so they don't need to be in `pyproject.toml`, but the network must be accessible from the compute node.

---

## Extension Pattern Summary

To add a new experiment variant:

```
1. src/<component>/my_impl.py       ← subclass Base<Component>, implement abstract methods
2. configs/<component>/my_impl.yaml ← _target_ + kwargs
3. src/experiment/run.py            ← fill in main loop logic
```

That is the full surface area. `main.py` and the utils do not need to change for new experiments.
