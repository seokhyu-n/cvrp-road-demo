import numpy as np
import pytest

pytest.importorskip("ortools")

from cvrp.ortools_baseline import solve_ortools  # noqa: E402
from cvrp.solver import distance_matrix, nearest_neighbor_trips, trips_length  # noqa: E402


def _instance(seed, n=12):
    rng = np.random.default_rng(seed)
    xy = rng.random((n + 1, 2))
    demands = [0] + [int(d) for d in rng.integers(1, 6, n)]
    return distance_matrix(xy), demands, 12


def test_ortools_is_feasible_and_covers_everyone():
    dist, demands, cap = _instance(0)
    trips = solve_ortools(dist, demands, cap, time_limit_s=0.3)
    assert all(t[0] == t[-1] == 0 for t in trips)
    assert sorted(c for t in trips for c in t if c) == list(range(1, len(demands)))
    assert all(sum(demands[c] for c in t) <= cap for t in trips)


def test_ortools_not_worse_than_nearest_neighbor():
    for seed in range(3):
        dist, demands, cap = _instance(seed)
        ort = trips_length(solve_ortools(dist, demands, cap, time_limit_s=0.3), dist)
        nn = trips_length(nearest_neighbor_trips(dist, demands, cap), dist)
        assert ort <= nn + 1e-9


def test_ortools_rejects_fractional_demands():
    dist, _, _ = _instance(0, n=3)
    with pytest.raises(ValueError):
        solve_ortools(dist, [0, 1.5, 1, 1], 5, time_limit_s=0.1)
