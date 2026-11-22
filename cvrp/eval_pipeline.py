"""
데모 추론 파이프라인의 단계별 효과 측정.

실제 사용 상황처럼 '위경도' 인스턴스를 만든다:
  - 서울 근처 약 11km × 11km 영역에 고객을 균일 배치
  - 창고(depot)는 영역 안 임의 위치 (학습 때처럼 중앙에 있지 않음)

비교 대상 (거리 = cos(위도) 보정 평면 직선거리, km 근사):
  NN            : 용량 제약 Nearest Neighbor (웹 UI의 Baseline)
  NN+2opt       : NN에 같은 2-opt를 적용한 공정 비교용 Baseline
  A legacy      : 기존 데모 (min-max 정규화 + greedy)
  B +norm       : depot 중앙 정규화 + greedy
  C +best-of-N  : B + sampling N개 중 최선
  D +2opt       : C + 회차별 2-opt (현재 데모 기본값)
  OR-Tools      : Routing solver + Guided Local Search (시간 제한, 최적에 가까운 참조해)

gap = 인스턴스별 (method - OR-Tools)/OR-Tools 평균. 0%에 가까울수록 참조해에 가깝다.

--metric road 이면 모든 방법을 OSRM 도로 거리(km)로 평가한다 (eval_set 이름 끝에 -road).
  A~D는 지금처럼 직선거리로 경로를 고른 뒤 도로 거리로 잰다.
  E는 D와 같지만 후보 선택·2-opt를 도로 거리로 한다 (도로 거리 반영 효과).
  NN·NN+2opt·OR-Tools는 도로 거리 행렬로 푼다.
  도로 행렬은 results/road_cache/에 캐시된다 (처음 한 번만 OSRM 호출, 초당 1회).
OR-Tools는 평가 전용 의존성이다 (pip install -r requirements-dev.txt). 없으면 건너뛴다.

usage:
  python -m cvrp.eval_pipeline --instances 200 --customers 20 --samples 32 --note "무엇을 바꿨는지"

실행할 때마다 results/runs.csv 에 Run 단위로 결과가 쌓인다 (--no-log 로 끄기).
기록 보기: python -m cvrp.runlog
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cvrp.model import CVRPModel
from cvrp.solver import (
    KM_PER_DEG_LAT,
    distance_matrix,
    nearest_neighbor_trips,
    solve_instance,
    to_planar,
    trips_length,
    two_opt,
)
from cvrp.road import RoadCache
from cvrp.runlog import append_run, git_commit, next_run_id

REPO = Path(__file__).resolve().parents[1]


def make_instance(rng: np.random.Generator, n_customers: int):
    lat_lo, lat_hi = 37.50, 37.60
    lng_lo, lng_hi = 126.92, 127.045  # cos(37.55°)≈0.79 → 위도 폭과 비슷한 실제 거리
    lat = rng.uniform(lat_lo, lat_hi, n_customers + 1)
    lng = rng.uniform(lng_lo, lng_hi, n_customers + 1)
    coords = np.stack([lat, lng], axis=1).tolist()
    capacity = float(rng.integers(20, 51))
    demands = [0.0] + rng.integers(1, 10, n_customers).astype(float).tolist()
    return coords, demands, capacity


def eval_set_id(instances: int, customers: int, seed: int) -> str:
    """평가셋 식별자. 이 값이 같으면 완전히 같은 인스턴스 묶음이다."""
    return f"seoul11km-n{customers}-x{instances}-s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--customers", type=int, default=20)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--note", default="", help="이번 Run에서 바꾼 것 (runs.csv에 기록)")
    ap.add_argument("--log", default=str(REPO / "results" / "runs.csv"))
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--ortools-time", type=float, default=1.0,
                    help="OR-Tools 인스턴스당 시간 제한(초). 0이면 건너뜀")
    ap.add_argument("--metric", choices=["planar", "road"], default="planar",
                    help="평가 거리: planar(직선, 기본) 또는 road(OSRM 도로 거리)")
    args = ap.parse_args()
    road = args.metric == "road"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = CVRPModel(embed_dim=128, n_heads=8, n_layers=3)
    model.load_state_dict(ckpt["model"])
    model.eval()

    S = args.samples
    # name -> (solve_instance kwargs, 기록용 설정)
    variants = {
        "A legacy (minmax+greedy)": dict(normalize="minmax", n_samples=0, use_two_opt=False),
        "B +depot-center norm": dict(normalize="depot_center", n_samples=0, use_two_opt=False),
        f"C +best-of-{S}": dict(normalize="depot_center", n_samples=S, use_two_opt=False),
        "D +2-opt (new default)": dict(normalize="depot_center", n_samples=S, use_two_opt=True),
    }
    road_aware = set()
    if road:
        e_name = "E +road-aware select/2-opt"
        variants[e_name] = dict(normalize="depot_center", n_samples=S, use_two_opt=True)
        road_aware.add(e_name)

    use_ort = args.ortools_time > 0
    if use_ort:
        try:
            from cvrp.ortools_baseline import solve_ortools
        except ImportError:
            print("[WARN] ortools not installed → OR-Tools 비교를 건너뜁니다 (pip install -r requirements-dev.txt)")
            use_ort = False
    ort_name = f"OR-Tools (GLS {args.ortools_time:g}s)"

    base_eset = eval_set_id(args.instances, args.customers, args.seed)
    cache = RoadCache(REPO / "results" / "road_cache" / f"{base_eset}.npz") if road else None

    rng = np.random.default_rng(args.seed)
    lengths = {k: [] for k in ["NN", "NN+2opt", *variants, *([ort_name] if use_ort else [])]}
    times = {k: 0.0 for k in lengths}

    for i in range(args.instances):
        coords, demands, cap = make_instance(rng, args.customers)
        planar = distance_matrix(to_planar(coords))
        if road:
            dist_m, _ = cache.get(i, coords)
            road_km = dist_m / 1000.0
            dist = road_km  # 평가 거리 (km)
            if cache.dirty and (i + 1) % 20 == 0:
                cache.save()  # 중간 저장: 끊겨도 받은 만큼은 남긴다
        else:
            road_km = None
            dist = planar  # 평가 거리 (위도 '도' 단위)

        t0 = time.perf_counter()
        nn = nearest_neighbor_trips(dist, demands, cap)
        t_nn = time.perf_counter() - t0
        times["NN"] += t_nn
        lengths["NN"].append(trips_length(nn, dist))

        t0 = time.perf_counter()
        nn2 = [two_opt(t, dist) for t in nn]
        times["NN+2opt"] += t_nn + (time.perf_counter() - t0)  # NN 생성 + 2-opt
        lengths["NN+2opt"].append(trips_length(nn2, dist))

        for name, kw in variants.items():
            t0 = time.perf_counter()
            sel = dict(dist=road_km) if name in road_aware else {}
            res = solve_instance(model, coords, demands, cap, seed=i, **kw, **sel)
            times[name] += time.perf_counter() - t0
            lengths[name].append(trips_length(res["trips"], dist))

        if use_ort:
            t0 = time.perf_counter()
            ort = solve_ortools(dist, demands, cap, time_limit_s=args.ortools_time)
            times[ort_name] += time.perf_counter() - t0
            lengths[ort_name].append(trips_length(ort, dist))

    if road:
        cache.save()
    unit_km = 1.0 if road else KM_PER_DEG_LAT

    nn_arr = np.array(lengths["NN"])
    settings = {
        "NN": dict(normalize="", n_samples="", two_opt=False),
        "NN+2opt": dict(normalize="", n_samples="", two_opt=True),
        **{
            k: dict(normalize=v["normalize"], n_samples=v["n_samples"], two_opt=v["use_two_opt"])
            for k, v in variants.items()
        },
        ort_name: dict(normalize="", n_samples="", two_opt=False),
    }
    ort_arr = np.array(lengths[ort_name]) if use_ort else None
    rows = []
    for name, vals in lengths.items():
        v = np.array(vals)
        imp = (nn_arr - v) / nn_arr * 100
        rows.append({
            "method": name,
            **settings[name],
            "mean_km": round(float(v.mean() * unit_km), 2),
            "vs_nn_pct": round(float(imp.mean()), 2),
            "win_pct": round(float((imp > 0).mean() * 100), 1),
            "worst_pct": round(float(imp.min()), 1),
            "ms_per_inst": round(times[name] / args.instances * 1000, 2),
            "gap_ort_pct": round(float(((v - ort_arr) / ort_arr * 100).mean()), 2) if use_ort else "",
        })

    eset = base_eset + ("-road" if road else "")
    print(f"\neval_set={eset}  samples={S}  metric={args.metric}")
    print(f"{'method':<28}{'mean km':>9}{'vs NN':>9}{'win%':>7}{'worst':>9}{'gap':>9}{'ms/inst':>10}")
    for r in rows:
        gap = f"{r['gap_ort_pct']:+8.2f}%" if use_ort else f"{'-':>9}"
        print(
            f"{r['method']:<28}{r['mean_km']:9.2f}{r['vs_nn_pct']:+8.2f}%"
            f"{r['win_pct']:6.0f}%{r['worst_pct']:+8.1f}%{gap}{r['ms_per_inst']:10.2f}"
        )
    print("\nvs NN = 인스턴스별 (NN - method)/NN 평균, win% = NN보다 짧았던 비율, worst = 가장 나빴던 인스턴스")
    if use_ort:
        print("gap = 인스턴스별 (method - OR-Tools)/OR-Tools 평균 (0%에 가까울수록 좋음)")

    if not args.no_log:
        log = Path(args.log)
        run_id = next_run_id(log)
        append_run(log, rows, run_id, {
            "git_commit": git_commit(REPO),
            "ckpt": Path(args.ckpt).name,
            "eval_set": eset,
            "note": args.note,
        })
        print(f"\n[{run_id}] {len(rows)} rows → {log}")


if __name__ == "__main__":
    main()
