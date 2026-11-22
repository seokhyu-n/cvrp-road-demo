import numpy as np
import pytest

app_module = pytest.importorskip("api.app")
from fastapi.testclient import TestClient  # noqa: E402

if app_module.model is None:
    pytest.skip("checkpoint not found", allow_module_level=True)

client = TestClient(app_module.app)

OSRM_CALLS = []


def _fake_osrm(coords, **kwargs):
    """네트워크 없이 도로 행렬 흉내: 직선거리의 1.3배 + 방향별로 조금 다름 (m)."""
    from cvrp.solver import distance_matrix, to_planar

    OSRM_CALLS.append(len(coords))
    base_m = distance_matrix(to_planar(coords)) * 111_320
    n = len(coords)
    skew = 1.0 + 0.1 * (np.arange(n)[:, None] > np.arange(n)[None, :])
    return base_m * 1.3 * skew, base_m / 8.0


@pytest.fixture(autouse=True)
def no_real_osrm(monkeypatch):
    OSRM_CALLS.clear()
    monkeypatch.setattr(app_module, "fetch_osrm_table", _fake_osrm)


def _payload(n_customers, n_samples=8, seed=0):
    rng = np.random.default_rng(seed)
    coords = [[37.55 + rng.random() * 0.1, 126.95 + rng.random() * 0.12] for _ in range(n_customers + 1)]
    demands = [0] + [1] * n_customers
    return {"coords": coords, "demands": demands, "capacity": 10, "n_samples": n_samples}


def test_health_reports_threads_and_hides_key_length():
    r = client.get("/health").json()
    assert r["torch_threads"] >= 1
    assert "kakao_key_len" not in r


def test_solve_ok():
    r = client.post("/solve", json=_payload(10)).json()
    assert "error" not in r
    assert sorted(c for t in r["trips"] for c in t if c) == list(range(1, 11))


def test_road_metric_is_default_and_uses_osrm_once():
    r = client.post("/solve", json=_payload(10)).json()
    assert r["settings"]["metric_used"] == "road"
    assert r["road_fallback"] is None
    assert OSRM_CALLS == [11]


def test_straight_metric_does_not_call_osrm():
    r = client.post("/solve", json={**_payload(10), "metric": "straight"}).json()
    assert r["settings"]["metric_used"] == "straight"
    assert OSRM_CALLS == []


def test_road_falls_back_to_straight_when_osrm_fails(monkeypatch):
    def boom(coords, **kwargs):
        raise RuntimeError("timeout")

    monkeypatch.setattr(app_module, "fetch_osrm_table", boom)
    r = client.post("/solve", json=_payload(10)).json()
    assert "error" not in r
    assert r["settings"]["metric_used"] == "straight"
    assert "timeout" in r["road_fallback"]
    assert sorted(c for t in r["trips"] for c in t if c) == list(range(1, 11))


def test_rejects_unknown_metric():
    r = client.post("/solve", json={**_payload(5), "metric": "manhattan"}).json()
    assert "metric must be" in r["error"]


def test_greedy_decode_is_backward_compatible():
    body = {**_payload(5), "decode": "greedy"}
    assert client.post("/solve", json=body).json()["settings"]["n_samples"] == 0


def test_rejects_too_many_points():
    r = client.post("/solve", json=_payload(app_module.MAX_NODES, n_samples=0)).json()
    assert "too many points" in r["error"]


def test_rejects_large_samples_times_points():
    r = client.post("/solve", json=_payload(100, n_samples=app_module.MAX_SAMPLES)).json()
    assert "n_samples × points" in r["error"]


def test_rejects_out_of_range_samples():
    r = client.post("/solve", json=_payload(5, n_samples=app_module.MAX_SAMPLES + 1)).json()
    assert "n_samples must be" in r["error"]
