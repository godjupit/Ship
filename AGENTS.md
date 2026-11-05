# Repository Guidelines

## Project Structure & Module Organization

- `supership/` contains inference, submission validation, benchmarking, and scoring entrypoints.
- `engine/` implements models, datasets, transforms, and solvers. Model forward execution starts in `engine/rtv4/rtv4.py`; HGNetv2 lives in `engine/backbone/`.
- `configs/` contains YAML configurations with shared includes. SuperShip inference uses `configs/dfine/dfine_hgnetv2_s_supership.yml` and `val_dataloader` settings.
- `tests/` contains automated tests; `scripts/` contains profiling and analysis utilities.
- `data/`, `weights/`, `output/`, and `analysis/` hold local datasets, checkpoints, results, and profiling artifacts. Keep generated assets out of commits.

## Build, Test, and Development Commands

Run commands from the repository root:

```bash
source activate_env.sh
python -m unittest discover -s tests -v
PYTHON_BIN=.venv/bin/python RUNS=3 bash ./benchmark_reference.sh
bash ./score_reference.sh output/reference_submission.json
python scripts/profile_inference.py --runs 3 --batch-size 8
```

These activate the environment, run tests, measure end-to-end inference, evaluate accuracy, and collect stage timings. See `ENVIRONMENT.md` and `requirements-cuda124.txt` for CUDA dependencies. Benchmarks require local reference data and `weights/baseline.pth`; training additionally requires training data.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `PascalCase` classes. Follow surrounding YAML formatting and existing registry conventions. Add brief function explanations and comments at important synchronization or data-transfer operations. No repository-wide formatter or linter is configured; check patches with `git diff --check`.

## Testing Guidelines

Use `unittest`, naming files `test_*.py` and methods `test_*`. Test output ordering, serialization, partial batches, and CPU compatibility when changing inference. Gate CUDA-specific tests on device availability. No numerical coverage threshold is configured.

For performance changes, use `bash ./benchmark_reference.sh`, warm up first, compare repeated-run medians under identical settings, and check output equality. Save results separately: default benchmark outputs are overwritten. Keep profiling scripts consistent with production inference.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, such as “Restore baseline inference transfers.” Commit each validated functional change. Include test commands, parameters, measured times, medians, output checks, and limitations in the commit body. Preserve unrelated user edits. Treat `note.md` as user-maintained; do not add progress, experiment records, or summaries unless explicitly requested.

PR descriptions should explain the problem, final behavior, validation, and performance evidence. Link relevant issues when available. Never claim acceleration from a single noisy run. Current validated inference uses TensorRT, CPU YUV decoding with a PGO9 JPEG library, and GPU fused preprocessing. Previously reverted optimizations require fresh evidence before reintroduction.

https://lab.cs.tsinghua.edu.cn/scc/Applications/SuperShip/
这个是我赛题要求


关于赛事规则、赛题、计分与提交指南等比赛详细内容请届时点击下方链接查看：

https://lab.cs.tsinghua.edu.cn/scc
