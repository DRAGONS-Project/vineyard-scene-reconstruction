# [Project name]

> Full-stack experiment template. Stack: **Hydra** (config) · **W&B** (logging) · **PyTorch** · **uv** · **SLURM**.

## 🛠️ Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Change `name = "project"` in `pyproject.toml`.
3. Set the output dir in `configs/hydra/default.yaml` and the W&B `project` in `configs/logger/wandb/default.yaml`.
4. Run:
   ```sh
   uv sync
   uv run pre-commit install
   ```

## 🚀 How to use it

```sh
# Local run
WANDB_MODE=offline uv run python src/main.py

# SLURM
bash slurm/SLURM_script.sh --script slurm/run_experiment.sh --time 04:00:00 --mem 32GB --gpu 1
```

To build your experiment:
1. Subclass `BaseModel` → `src/model/`, add a matching YAML in `configs/model/`.
2. Subclass `BaseDataset` → `src/dataset/`, add a matching YAML in `configs/dataset/`.
3. Subclass `BaseMetric` → `src/metric/`, register in `configs/metric/default.yaml`.
4. Fill in the loop in `src/experiment/run.py`.

Log via the namespaced logger:
```python
logger.log({"metrics/loss": 0.42, "metadata/preds": tensor})
#           ^-- wandb scalar      ^-- saved as CSV in run dir
```

## 📚 Citation

> Citation instructions. How people should cite your work?
