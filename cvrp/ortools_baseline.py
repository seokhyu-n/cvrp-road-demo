"""
OR-Tools 기반 CVRP 참조해(reference) — 평가 전용 (데모 서버에는 필요 없음)

Routing solver + Guided Local Search를 시간 제한만큼 돌려 최적에 가까운 해를 구한다.
모델 결과를 "최적해 대비 몇 % 차이(gap)"로 말하기 위한 기준선이다.
"""
from typing import List, Sequence

import numpy as np

DIST_SCALE = 1_000_000  # 정수 거리로 바꿀 배율 (평면 좌표 '도' 단위 → 약 0.1m 해상도)


def solve_ortools(
    dist: np.ndarray,
    demands: Sequence[float],
    capacity: float,
    time_limit_s: float = 1.0,
) -> List[List[int]]:
    """
    dist: (N, N) 거리 행렬 (0 = depot)
    returns: trips, 각 trip은 [0, ..., 0]
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = len(demands)
    if any(float(d) != int(d) for d in demands) or float(capacity) != int(capacity):
        raise ValueError("OR-Tools baseline expects integer demands/capacity")

    n_vehicles = n - 1  # 고객 한 명씩 실어도 되는 상한 (안 쓰는 차량은 비용 0)
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    dist_int = np.rint(dist * DIST_SCALE).astype(np.int64)

    def dist_cb(i, j):
        return int(dist_int[manager.IndexToNode(i), manager.IndexToNode(j)])

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    dem = [int(d) for d in demands]

    def demand_cb(i):
        return dem[manager.IndexToNode(i)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [int(capacity)] * n_vehicles, True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(max(1, int(time_limit_s * 1000)))

    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("OR-Tools found no solution")

    trips: List[List[int]] = []
    for v in range(n_vehicles):
        idx = routing.Start(v)
        trip = [0]
        idx = sol.Value(routing.NextVar(idx))
        while not routing.IsEnd(idx):
            trip.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        if len(trip) > 1:
            trip.append(0)
            trips.append(trip)
    return trips
