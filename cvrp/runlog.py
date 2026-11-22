"""
평가 Run 기록 (results/runs.csv)

같은 평가셋(eval_set)에서 한 번에 한 변수씩 바꿔 가며 실행한 결과를
Run 단위로 쌓는다. 한 Run 안의 방법(method)마다 한 줄씩 기록한다.
"""
import csv
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

FIELDS = [
    "run_id",
    "date",
    "git_commit",
    "ckpt",
    "eval_set",
    "method",
    "normalize",
    "n_samples",
    "two_opt",
    "mean_km",
    "vs_nn_pct",
    "win_pct",
    "worst_pct",
    "ms_per_inst",
    "note",
    "gap_ort_pct",  # OR-Tools 참조해 대비 (Run-03부터)
]

_RUN_RE = re.compile(r"^Run-(\d+)$")


def next_run_id(path: Path) -> str:
    """기존 기록의 가장 큰 Run 번호 + 1 (파일이 없으면 Run-01)."""
    last = 0
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:  # BOM 있어도 헤더 인식
            for row in csv.DictReader(f):
                m = _RUN_RE.match(row.get("run_id", ""))
                if m:
                    last = max(last, int(m.group(1)))
    return f"Run-{last + 1:02d}"


def git_commit(repo: Path, ignore: Iterable[str] = ("results/runs.csv",)) -> str:
    """
    현재 커밋 short hash. 커밋 안 된 변경이 있으면 '-dirty'를 붙인다.
    기록 파일 자체(runs.csv)는 Run마다 바뀌므로 dirty 판정에서 뺀다.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        excludes = [f":(exclude){p}" for p in ignore]
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", *excludes],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _migrate_header(path: Path) -> None:
    """기존 파일 헤더가 FIELDS와 다르면(새 컬럼 추가 등) 기존 행을 보존한 채 헤더를 맞춘다."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == FIELDS:
            return
        old_rows = list(reader)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in old_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def append_run(path: Path, rows: Iterable[Dict], run_id: str, common: Dict) -> List[Dict]:
    """rows(방법별 결과)에 run_id·공통 정보를 붙여 CSV에 추가. 헤더는 새 파일일 때만 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        _migrate_header(path)
    new_file = not path.exists() or path.stat().st_size == 0
    date = common.get("date") or datetime.now().strftime("%Y-%m-%d %H:%M")
    out = []
    # utf-8-sig: 엑셀에서 열어도 한글 note가 깨지지 않게
    with path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for r in rows:
            full = {**common, **r, "run_id": run_id, "date": date}
            w.writerow(full)
            out.append(full)
    return out


def show(path: Path, eval_set: str = "", method: str = "") -> None:
    """기록을 Run 순서대로 출력. eval_set / method(부분 문자열)로 거를 수 있다."""
    if not path.exists():
        print(f"no runs yet: {path}")
        return
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if eval_set in r["eval_set"] and method in r["method"]
        ]
    # 평가셋별로 묶고, 그 안에서는 기록 순서 유지 (sort는 안정 정렬)
    rows.sort(key=lambda r: r["eval_set"])
    cur_set = None
    for r in rows:
        if r["eval_set"] != cur_set:
            cur_set = r["eval_set"]
            print(f"\n== {cur_set} ==")
            print(f"{'run':<8}{'commit':<15}{'method':<28}{'mean km':>9}{'vs NN':>9}{'win%':>7}{'worst':>8}{'gap':>9}{'ms':>9}  note")
        gap = r.get("gap_ort_pct") or ""
        gap_s = f"{float(gap):+8.2f}%" if gap else f"{'-':>9}"
        print(
            f"{r['run_id']:<8}{r['git_commit']:<15}{r['method']:<28}{float(r['mean_km']):9.2f}"
            f"{float(r['vs_nn_pct']):+8.2f}%{float(r['win_pct']):6.0f}%{float(r['worst_pct']):+7.1f}%"
            f"{gap_s}{float(r['ms_per_inst']):9.1f}  {r['note']}"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="results/runs.csv 보기")
    ap.add_argument("--log", default=str(Path(__file__).resolve().parents[1] / "results" / "runs.csv"))
    ap.add_argument("--eval-set", default="", help="예: n20")
    ap.add_argument("--method", default="", help="예: D")
    a = ap.parse_args()
    show(Path(a.log), a.eval_set, a.method)
