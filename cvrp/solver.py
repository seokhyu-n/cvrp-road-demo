# cvrp/solver.py
"""
추론 파이프라인 (API / 평가 스크립트 공용)

  위경도 → 평면 좌표 → 모델 입력 정규화 → 디코딩(greedy + best-of-N 샘플링)
  → capacity-only 회차 분할 → 회차별 2-opt → 가장 짧은 후보 선택
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

KM_PER_DEG_LAT = 111.32  # 위도 1도 ≈ 111.32km

NORMALIZE_MODES = ("depot_center", "minmax")


# -----------------------------
# 좌표 변환 / 정규화
# -----------------------------
def to_planar(coords_latlng: Sequence[Sequence[float]]) -> np.ndarray:
    """
    [lat, lng] → 평면 좌표 [x, y] (단위: 위도 '도').
    경도 1도의 실제 길이는 cos(위도)배로 짧아지므로 x에 cos(lat0)를 곱한다.
    (lat0 = 전체 지점의 평균 위도, 도시 규모에선 오차 무시 가능)
    """
    arr = np.asarray(coords_latlng, dtype=np.float64)
    lat0 = math.radians(float(arr[:, 0].mean()))
    x = arr[:, 1] * math.cos(lat0)
    y = arr[:, 0]
    return np.stack([x, y], axis=1)


def normalize_minmax(xy: np.ndarray) -> np.ndarray:
    """(기존 방식) 전체 bbox 기준 min-max. depot 위치는 아무 데나 올 수 있음."""
    mins = xy.min(axis=0)
    span = max(float((xy.max(axis=0) - mins).max()), 1e-9)
    return (xy - mins) / span


def normalize_depot_center(xy: np.ndarray) -> np.ndarray:
    """
    학습 분포(depot=(0.5,0.5), 고객 ⊂ [0,1]^2)에 맞춘 정규화.
    depot을 중앙에 두고, depot에서 가장 먼 축 방향 거리가 0.5가 되도록 균일 스케일.
    """
    offset = xy - xy[0]
    half = max(float(np.abs(offset).max()), 1e-9)
    return offset / (2.0 * half) + 0.5


def normalize_for_model(xy: np.ndarray, mode: str = "depot_center") -> np.ndarray:
    if mode == "depot_center":
        return normalize_depot_center(xy)
    if mode == "minmax":
        return normalize_minmax(xy)
    raise ValueError(f"normalize must be one of {NORMALIZE_MODES}")


# -----------------------------
# 거리 / 회차
# -----------------------------
def distance_matrix(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def trip_length(trip: Sequence[int], dist: np.ndarray) -> float:
    return float(sum(dist[trip[i], trip[i + 1]] for i in range(len(trip) - 1)))


def trips_length(trips: Sequence[Sequence[int]], dist: np.ndarray) -> float:
    return sum(trip_length(t, dist) for t in trips)


def extract_order(path: Sequence[int], n_nodes: int) -> List[int]:
    """모델 path에서 고객 방문 순서만 추출 (depot·중복·범위 밖 제거, 누락 고객은 뒤에 추가)."""
    order: List[int] = []
    seen = {0}
    for v in path:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= iv < n_nodes and iv not in seen:
            order.append(iv)
            seen.add(iv)
    order.extend(i for i in range(1, n_nodes) if i not in seen)
    return order


def split_trips_capacity_only(
    order: Sequence[int],
    demands: Sequence[float],
    capacity: float,
) -> List[List[int]]:
    """방문 순서대로 싣다가 load + demand > capacity 가 되는 순간에만 depot 복귀."""
    trips: List[List[int]] = []
    cur = [0]
    load = 0.0
    for c in order:
        d = float(demands[c])
        if d > capacity:
            raise ValueError(f"Customer {c} demand({d}) > capacity({capacity})")
        if load + d > capacity and len(cur) > 1:
            cur.append(0)
            trips.append(cur)
            cur = [0]
            load = 0.0
        cur.append(c)
        load += d
    cur.append(0)
    trips.append(cur)
    return trips


def trip_load(trip: Sequence[int], demands: Sequence[float]) -> float:
    return sum(float(demands[i]) for i in trip if i != 0)


def is_symmetric(dist: np.ndarray) -> bool:
    return bool(np.allclose(dist, dist.T))


def two_opt(
    trip: Sequence[int],
    dist: np.ndarray,
    eps: float = 1e-12,
    symmetric: Optional[bool] = None,
) -> List[int]:
    """
    한 회차(depot 시작·끝 고정) 안에서 2-opt.
    구간 [i..j]를 뒤집었을 때 짧아지면 적용, 개선이 없을 때까지 반복.
    회차 안의 고객 집합은 그대로라 적재량(용량 제약)은 변하지 않는다.

    도로 거리처럼 비대칭 행렬이면 뒤집힌 구간 내부 간선의 방향도 바뀌므로
    그 차이까지 더해 판단한다 (symmetric=None이면 행렬을 보고 판단).
    """
    t = list(trip)
    if symmetric is None:
        symmetric = is_symmetric(dist)
    if symmetric and len(t) <= 4:  # [0, a, b, 0] 은 뒤집어도 길이 동일 (대칭일 때만)
        return t
    improved = True
    while improved:
        improved = False
        for i in range(1, len(t) - 2):
            for j in range(i + 1, len(t) - 1):
                a, b, c, d = t[i - 1], t[i], t[j], t[j + 1]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if not symmetric:
                    seg = t[i : j + 1]
                    fwd = sum(dist[seg[k], seg[k + 1]] for k in range(len(seg) - 1))
                    rev = sum(dist[seg[k + 1], seg[k]] for k in range(len(seg) - 1))
                    delta += rev - fwd
                if delta < -eps:
                    t[i : j + 1] = reversed(t[i : j + 1])
                    improved = True
    return t


# -----------------------------
# Baseline
# -----------------------------
def nearest_neighbor_trips(
    dist: np.ndarray, demands: Sequence[float], capacity: float
) -> List[List[int]]:
    """용량 제약 Nearest Neighbor (웹 UI의 Baseline과 같은 규칙)."""
    n = len(demands)
    unvisited = set(range(1, n))
    trips: List[List[int]] = []
    cur_trip, cur, load = [0], 0, 0.0
    while unvisited:
        feas = [i for i in unvisited if load + float(demands[i]) <= capacity]
        if not feas:
            cur_trip.append(0)
            trips.append(cur_trip)
            cur_trip, cur, load = [0], 0, 0.0
            continue
        nxt = min(feas, key=lambda i: dist[cur, i])
        cur_trip.append(nxt)
        load += float(demands[nxt])
        unvisited.remove(nxt)
        cur = nxt
    cur_trip.append(0)
    trips.append(cur_trip)
    return trips


# -----------------------------
# 전체 추론
# -----------------------------
def _candidate(path: Sequence[int], n: int, demands, capacity, dist, use_two_opt, symmetric=None):
    order = extract_order(path, n)
    trips = split_trips_capacity_only(order, demands, capacity)
    raw_len = trips_length(trips, dist)
    if use_two_opt:
        trips = [two_opt(t, dist, symmetric=symmetric) for t in trips]
    return {
        "order": order,
        "trips": trips,
        "raw_length": raw_len,
        "length": trips_length(trips, dist),
    }


@torch.no_grad()
def solve_instance(
    model: torch.nn.Module,
    coords_latlng: Sequence[Sequence[float]],
    demands: Sequence[float],
    capacity: float,
    n_samples: int = 32,
    use_two_opt: bool = True,
    normalize: str = "depot_center",
    seed: int = 0,
    device: str = "cpu",
    dist: Optional[np.ndarray] = None,
    km_per_unit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    greedy 1개 + sampling n_samples개 후보를 만들고, 각 후보를 회차 분할(+2-opt)한 뒤
    가장 짧은 후보를 반환한다.

    dist를 주지 않으면 평면 직선거리(단위 '위도 도', km_per_unit=111.32)로 고르고,
    도로 거리 행렬(km, 비대칭 가능)을 주면 그 거리로 후보 선택과 2-opt를 한다.
    모델 입력은 어느 경우든 좌표다.
    """
    n = len(demands)
    xy = to_planar(coords_latlng)
    if dist is None:
        dist = distance_matrix(xy)
        km_per_unit = KM_PER_DEG_LAT if km_per_unit is None else km_per_unit
    else:
        dist = np.asarray(dist, dtype=np.float64)
        km_per_unit = 1.0 if km_per_unit is None else km_per_unit
    symmetric = is_symmetric(dist)
    xy_model = normalize_for_model(xy, normalize)

    coords_t = torch.tensor(xy_model, dtype=torch.float32, device=device).unsqueeze(0)
    demands_t = torch.tensor(list(demands), dtype=torch.float32, device=device).unsqueeze(0)
    cap_t = torch.tensor([float(capacity)], dtype=torch.float32, device=device)

    greedy_path, _ = model(coords_t, demands_t, cap_t, decode_type="greedy")
    greedy = _candidate(greedy_path[0].tolist(), n, demands, capacity, dist, use_two_opt, symmetric)
    candidates = [greedy]

    if n_samples > 0:
        # 같은 입력 → 같은 결과가 나오도록 전역 RNG를 건드리지 않고 시드 고정
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            paths, _ = model(
                coords_t.expand(n_samples, -1, -1),
                demands_t.expand(n_samples, -1),
                cap_t.expand(n_samples),
                decode_type="sampling",
            )
        for p in paths.tolist():
            candidates.append(_candidate(p, n, demands, capacity, dist, use_two_opt, symmetric))

    best = min(candidates, key=lambda c: c["length"])
    best_raw = min(c["raw_length"] for c in candidates)

    to_km = lambda v: v * km_per_unit
    return {
        "visit_order": best["order"],
        "trips": best["trips"],
        "trip_loads": [trip_load(t, demands) for t in best["trips"]],
        "n_candidates": len(candidates),
        "stats": {
            "greedy_km": to_km(greedy["raw_length"]),
            "best_of_n_km": to_km(best_raw),
            "final_km": to_km(best["length"]),
        },
        "length": best["length"],
    }
