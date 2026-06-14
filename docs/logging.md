# Logging Structure

This document describes exactly what is logged, when, and where for every run of
this project. Read this before adding new log calls or changing metric names.

---

## 1. Directory layout

All run artefacts are written under `logs/` in the project root:

```
logs/
└── YYYY-MM-DD/
    └── HH-MM-SS/            ← one directory per run (Hydra output dir)
        ├── .hydra/           ← full resolved config snapshot (Hydra managed)
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── experiment.log    ← Python logging (INFO and above, written by Hydra)
        └── wandb/            ← W&B local artefacts
            └── offline-run-*/
                └── files/
                    └── metadata/
                        └── <name>.csv   ← per-run CSV files (see §2)
```

`config.exp.log_dir` is set to the Hydra output dir at startup by
`utils.hydra.preprocess_config()`. W&B's `dir=` argument points to the same path,
so W&B artefacts co-locate with Hydra artefacts.

If `run.dir` in `configs/hydra/default.yaml` is changed to an absolute path (e.g.
a cluster scratch disk), a symlink `logs/YYYY-MM-DD/HH-MM-SS →
<scratch>/YYYY-MM-DD/HH-MM-SS` is created automatically so the run is still
reachable from the project root.

---

## 2. W&B logger dispatch rules

`WandBLogger.log(log_dict)` routes keys by prefix:

| Key prefix   | Destination                                      |
|---|---|
| `metrics/*`  | `wandb.log()` scalar (prefix stripped before sending) |
| `metadata/*` | Appended row in `<run_dir>/wandb/.../files/<name>.csv` |
| anything else | **silently dropped** — do not use unprefixed keys |

Always include a step counter (e.g. `metrics/total_steps` or `metrics/epoch`) in
every `logger.log()` call so W&B can align curves from different runs on the same axis.

---

## 3. Run-specific logging

Document your project's logged metrics here. Example structure:

### Every N steps — training diagnostics

| W&B key (after prefix strip) | Description |
|---|---|
| `loss/train` | Training loss |
| `total_steps` | Global step counter (x-axis) |

### Final — aggregate metrics

| W&B key | Description |
|---|---|
| `metrics/<name>` | Result from `metric.compute_and_log()` for each registered metric |

---

## 4. W&B run organisation

| W&B field | Value |
|---|---|
| Project | Set in `configs/logger/wandb/default.yaml` |
| Group | Date subfolder, e.g. `2026-04-16` |
| Name | Time subfolder, e.g. `10-05-14` |
| Config | Full resolved Hydra config (all hyperparameters captured) |

Runs from the same experiment day are automatically grouped in the W&B UI.
To override the project or add an entity:

```bash
# Entity override
uv run python src/main.py logger.entity=your-team

# Custom run name (bypasses the time-based default)
uv run python src/main.py logger.name=my-run-label
```

---

## 5. What is NOT logged (and why)

| Item | Reason |
|---|---|
| W&B system metrics (CPU/GPU) | Enabled by default by W&B; not controlled here |

---

## 6. Adding a new metric

1. Create `src/metric/<name>.py` subclassing `BaseMetric`.
2. Implement `update(**kwargs)`, `compute_and_log() -> dict`, and `_reset()`.
   `_reset()` **must** be called at the top of `compute_and_log()` so successive
   calls each reflect only data accumulated since the previous call.
3. Add an entry to `configs/metric/default.yaml`:
   ```yaml
   my_metric:
     _target_: metric.<name>.<ClassName>
   ```
4. Call `metrics["my_metric"].update(...)` in the appropriate loop in `run.py`.
5. No changes needed to `run.py`'s boundary/final log calls — they iterate over
   `metrics.items()` automatically.
6. Document the new W&B keys in §3 above.
