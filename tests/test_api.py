import numpy as np
import pytest

app_module = pytest.importorskip("api.app")
from fastapi.testclient import TestClient  # noqa: E402

if app_module.model is None:
    pytest.skip("checkpoint not found", allow_module_level=True)

client = TestClient(app_module.app)


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
