import random

import numpy as np

from cvrp.road import RoadCache
from cvrp.solver import distance_matrix, trip_length, two_opt


def _asym(n, seed):
    rng = np.random.default_rng(seed)
    base = distance_matrix(rng.random((n, 2)))
    # 일방통행처럼 방향마다 0~40% 더 긴 거리
    return base * (1.0 + 0.4 * rng.random((n, n)))


def test_two_opt_asymmetric_never_worse_and_keeps_nodes():
    for seed in range(60):
        dist = _asym(9, seed)
        inner = list(range(1, 9))
        random.Random(seed).shuffle(inner)
        trip = [0] + inner + [0]
        out = two_opt(trip, dist)
        assert out[0] == out[-1] == 0
        assert sorted(out) == sorted(trip)
        assert trip_length(out, dist) <= trip_length(trip, dist) + 1e-9


def test_two_opt_asymmetric_can_reverse_two_customer_trip():
    # 0→1→2→0 보다 0→2→1→0 이 짧은 비대칭 행렬: 대칭 가정이면 놓친다
    dist = np.array([
        [0, 5, 1],
        [1, 0, 5],
        [5, 1, 0],
    ], dtype=float)
    out = two_opt([0, 1, 2, 0], dist)
    assert out == [0, 2, 1, 0]
    assert trip_length(out, dist) == 3


def test_road_cache_fetches_once_and_round_trips(tmp_path):
    calls = []

    def fake_fetch(coords):
        calls.append(len(coords))
        n = len(coords)
        d = np.arange(n * n, dtype=float).reshape(n, n) + 0.123456789
        return d, d * 2

    path = tmp_path / "cache.npz"
    coords = [[37.5, 127.0], [37.51, 127.01], [37.52, 127.02]]

    c1 = RoadCache(path, fetch=fake_fetch)
    d1, t1 = c1.get(0, coords)
    d1b, _ = c1.get(0, coords)
    assert calls == [3]
    assert d1 is d1b
    c1.save()

    c2 = RoadCache(path, fetch=fake_fetch)
    d2, t2 = c2.get(0, coords)
    assert calls == [3]  # 캐시에서 읽음
    np.testing.assert_array_equal(d1, d2)  # 처음 받은 값과 캐시 값이 완전히 같다
    np.testing.assert_array_equal(t1, t2)


def test_road_cache_refetches_when_size_differs(tmp_path):
    calls = []

    def fake_fetch(coords):
        calls.append(len(coords))
        n = len(coords)
        return np.ones((n, n)), np.ones((n, n))

    c = RoadCache(tmp_path / "c.npz", fetch=fake_fetch)
    c.get(0, [[0, 0], [1, 1]])
    c.get(0, [[0, 0], [1, 1], [2, 2]])
    assert calls == [2, 3]
