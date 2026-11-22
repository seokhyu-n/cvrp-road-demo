import random
from pathlib import Path

import numpy as np
import pytest
import torch

from cvrp.model import CVRPModel
from cvrp.solver import (
    distance_matrix,
    extract_order,
    nearest_neighbor_trips,
    normalize_depot_center,
    solve_instance,
    split_trips_capacity_only,
    to_planar,
    trip_length,
    trips_length,
    two_opt,
)

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "best.pt"


def test_to_planar_scales_longitude_by_cos_lat():
    # 위도 60도에선 경도 1도 = 위도 0.5도
    xy = to_planar([[60.0, 10.0], [60.0, 11.0]])
    assert xy[1, 0] - xy[0, 0] == pytest.approx(0.5, rel=1e-6)
    assert xy[0, 1] == xy[1, 1] == 60.0


def test_normalize_depot_center_puts_depot_at_center_and_inside_unit_box():
    xy = np.array([[3.0, 3.0], [0.0, 2.0], [7.0, 5.0], [4.0, -1.0]])
    out = normalize_depot_center(xy)
    assert out[0] == pytest.approx([0.5, 0.5])
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.abs(out - 0.5).max() == pytest.approx(0.5)


def test_extract_order_drops_depot_duplicates_and_appends_missing():
    assert extract_order([0, 3, 0, 3, 1, 9, 0], n_nodes=5) == [3, 1, 2, 4]


def test_split_respects_capacity_and_covers_all_customers():
    demands = [0, 4, 4, 4, 2]
    trips = split_trips_capacity_only([1, 2, 3, 4], demands, capacity=8)
    assert trips == [[0, 1, 2, 0], [0, 3, 4, 0]]


def test_split_rejects_oversized_demand():
    with pytest.raises(ValueError):
        split_trips_capacity_only([1], [0, 10], capacity=5)


def test_two_opt_untangles_crossing_and_keeps_nodes():
    # 정사각형을 X자로 도는 경로 → 2-opt 후 둘레를 돈다
    xy = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0.5]], dtype=float)
    dist = distance_matrix(xy)
    trip = [0, 3, 1, 2, 0]
    out = two_opt(trip, dist)
    assert out[0] == out[-1] == 0
    assert sorted(out) == sorted(trip)
    assert trip_length(out, dist) < trip_length(trip, dist)


def test_two_opt_never_worse_on_random_trips():
    rng = np.random.default_rng(0)
    for _ in range(50):
        xy = rng.random((9, 2))
        dist = distance_matrix(xy)
        inner = list(range(1, 9))
        random.Random(int(rng.integers(1e9))).shuffle(inner)
        trip = [0] + inner + [0]
        out = two_opt(trip, dist)
        assert sorted(out) == sorted(trip)
        assert trip_length(out, dist) <= trip_length(trip, dist) + 1e-12


def test_nearest_neighbor_respects_capacity():
    rng = np.random.default_rng(1)
    xy = rng.random((12, 2))
    demands = [0] + list(rng.integers(1, 6, 11))
    trips = nearest_neighbor_trips(distance_matrix(xy), demands, capacity=10)
    assert sorted(c for t in trips for c in t if c) == list(range(1, 12))
    assert all(sum(demands[c] for c in t) <= 10 for t in trips)


@pytest.fixture(scope="module")
def model():
    if not CKPT.exists():
        pytest.skip("checkpoint not found")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = CVRPModel(embed_dim=128, n_heads=8, n_layers=3)
    m.load_state_dict(ckpt["model"])
    return m.eval()


def _instance(seed=0, n=15):
    rng = np.random.default_rng(seed)
    coords = [[37.55 + rng.random() * 0.1, 126.95 + rng.random() * 0.12] for _ in range(n + 1)]
    demands = [0] + [int(d) for d in rng.integers(1, 9, n)]
    return coords, demands, 30.0


def test_solve_instance_is_feasible_and_deterministic(model):
    coords, demands, cap = _instance()
    a = solve_instance(model, coords, demands, cap, n_samples=8, seed=3)
    b = solve_instance(model, coords, demands, cap, n_samples=8, seed=3)
    assert a["trips"] == b["trips"]
    assert a["n_candidates"] == 9
    served = sorted(c for t in a["trips"] for c in t if c)
    assert served == list(range(1, len(demands)))
    assert all(load <= cap for load in a["trip_loads"])


def test_best_of_n_and_two_opt_never_worse_than_greedy(model):
    coords, demands, cap = _instance(seed=5)
    greedy = solve_instance(model, coords, demands, cap, n_samples=0, use_two_opt=False)
    best = solve_instance(model, coords, demands, cap, n_samples=16, use_two_opt=False)
    full = solve_instance(model, coords, demands, cap, n_samples=16, use_two_opt=True)
    assert best["length"] <= greedy["length"] + 1e-12
    assert full["length"] <= best["length"] + 1e-12
    s = full["stats"]
    assert s["final_km"] <= s["best_of_n_km"] <= s["greedy_km"] + 1e-9
