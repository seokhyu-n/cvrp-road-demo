# cvrp-road-demo

CVRP web demo: Transformer+Pointer model (PyTorch) → capacity-only trip split → per-trip 2-opt,
served by FastAPI (`api/app.py`) with a Leaflet UI (`web/index.html`), deployed on Render from `main`.

@docs/SPEC.md
@docs/STATE.md

## Environment (Windows)
- Use the project venv: `.\.venv\Scripts\python.exe`. Always `python -m pip ...`, never bare `pip`.
- First-time setup (CPU torch keeps it small):
  `python -m venv .venv` → `.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
  → `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`
- `requirements.txt` = what Render installs. Eval/test-only packages (ortools, pytest, httpx) go in `requirements-dev.txt`.

## Commands
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` (must pass before every commit)
- Server: `.\.venv\Scripts\python.exe -m uvicorn api.app:app --port 8000` (place search needs `KAKAO_REST_API_KEY`; without it, Nominatim fallback)
- Eval: `.\.venv\Scripts\python.exe -m cvrp.eval_pipeline --instances 200 --customers 20 --samples 32 --note "<what changed>"`
- View eval history: `.\.venv\Scripts\python.exe -m cvrp.runlog [--eval-set n20] [--method D]`

## Evaluation rules
- Every eval run appends rows to `results/runs.csv`. Never edit that file by hand.
- Compare only rows with the same `eval_set` (e.g. `seoul11km-n20-x200-s1234`). Keep the standard sets:
  n20 → `--instances 200 --customers 20`, n50 → `--instances 100 --customers 50`, both `--seed 1234`.
- Change one variable per run and describe it in `--note`.
- Commit code changes before recording a run so `git_commit` is clean (not `-dirty`).
- Distances are straight-line (cos(lat)-corrected planar, km approx.), not road distances.

## Code map (non-obvious parts)
- `cvrp/solver.py` is the single inference path used by both the API and the eval script. Change behavior there, not in `api/app.py`.
- Model input normalization must match training (`cvrp/data.py`: depot fixed at (0.5, 0.5), customers in [0,1]²) → `normalize_depot_center`.
- `/solve` has request caps (`MAX_NODES`, `MAX_SAMPLES`, `MAX_SAMPLE_NODES` in `api/app.py`) because the demo is public. Keep caps when changing it.
- Load checkpoints with `torch.load(..., weights_only=True)`.

## Git
- Work on a feature branch, open a PR to `main`; PRs are squash-merged. Merging to `main` deploys to Render.
- Commit messages: Conventional Commits prefix + Korean summary (e.g. `feat: ...`, `fix: ...`).

## Work-state management (do this without being asked)
- On session start, continue from "다음 할 일" in `docs/STATE.md`.
- After finishing each step, overwrite `docs/STATE.md` with the current state
  (지금 하는 일 · 다음 할 일 · 막힌 점 · 실행 방법), max 30 lines.
- If goals or done criteria change, update `docs/SPEC.md`.
- When a task is finished, prepend an entry to `docs/HISTORY.md` (date · 한 일 · 결정 · 다음 할 일).
  Do not import HISTORY.md with @; read it only when needed.
- `docs/NEXT_STEPS.md` holds the detailed improvement plan, background notes, and environment pitfalls.
  Read it when starting any item from "다음 할 일".
- Write these docs in Korean.
