# api/app.py
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import requests
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cvrp.model import CVRPModel
from cvrp.data import route_length

app = FastAPI(title="CVRP Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_INDEX = BASE_DIR / "web" / "index.html"
DEFAULT_MODEL_PATH = BASE_DIR / "checkpoints" / "best.pt"


class SolveRequest(BaseModel):
    coords: List[List[float]]   # [lat,lng]
    demands: List[float]        # demands[0]=0
    capacity: float
    decode: Optional[str] = "greedy"


class RouteRequest(BaseModel):
    coords: List[List[float]]   # [lat,lng] visiting order


def _normalize_uniform(coords_tensor: torch.Tensor):
    mins = coords_tensor.min(dim=0).values
    maxs = coords_tensor.max(dim=0).values
    span = max((maxs - mins).max().item(), 1e-9)
    return (coords_tensor - mins) / span


def _clean_path(path: List[int], n_nodes: int) -> List[int]:
    p = []
    for v in path:
        try:
            iv = int(v)
        except:
            continue
        if 0 <= iv < n_nodes:
            p.append(iv)

    if not p:
        return [0, 0]

    compact = [p[0]]
    for v in p[1:]:
        if v == 0 and compact[-1] == 0:
            continue
        compact.append(v)

    last_nonzero = -1
    for i in range(len(compact) - 1, -1, -1):
        if compact[i] != 0:
            last_nonzero = i
            break

    if last_nonzero == -1:
        return [0, 0]

    compact = compact[: last_nonzero + 1]

    if compact[0] != 0:
        compact = [0] + compact
    if compact[-1] != 0:
        compact = compact + [0]

    return compact


def split_trips_capacity_only(
    path: List[int],
    demands: List[float],
    capacity: float
) -> Tuple[List[List[int]], List[int]]:
    n = len(demands)

    order: List[int] = []
    seen = {0}
    for v in path:
        if v != 0 and (v not in seen) and (1 <= v < n):
            order.append(v)
            seen.add(v)

    for i in range(1, n):
        if i not in seen:
            order.append(i)

    trips: List[List[int]] = []
    cur = [0]
    load = 0.0

    for c in order:
        d = float(demands[c])
        if d > capacity:
            raise ValueError(f"Customer {c} demand({d}) > capacity({capacity})")

        if (load + d > capacity) and (len(cur) > 1):
            cur.append(0)
            trips.append(cur)
            cur = [0]
            load = 0.0

        cur.append(c)
        load += d

    cur.append(0)
    trips.append(cur)

    return trips, order


def _trip_load(trip: List[int], demands: List[float]) -> float:
    return sum(float(demands[i]) for i in trip if i != 0)


# -----------------------------
# ✅ Kakao geocode helpers
# -----------------------------
def _get_kakao_rest_key() -> str:
    # 요청 시점에 env를 읽는다(배포/재시작 이슈 줄이기)
    return (os.environ.get("KAKAO_REST_API_KEY", "") or "").strip()


def _kakao_headers() -> Dict[str, str]:
    key = _get_kakao_rest_key()
    return {"Authorization": f"KakaoAK {key}"}


def _kakao_keyword_search(query: str, limit: int = 30) -> List[Dict[str, Any]]:
    key = _get_kakao_rest_key()
    if not key:
        raise RuntimeError("KAKAO_REST_API_KEY is not set")

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    size = min(15, max(1, int(limit)))
    page = 1
    results: List[Dict[str, Any]] = []

    while len(results) < limit and page <= 45:
        params = {"query": query, "page": page, "size": size}
        r = requests.get(url, params=params, headers=_kakao_headers(), timeout=10)

        # ✅ 403/400 같은 에러면 이유(body)를 같이 내보내기
        if r.status_code != 200:
            raise RuntimeError(f"Kakao keyword search failed: {r.status_code} {r.text}")

        js = r.json()
        docs = js.get("documents", [])

        for d in docs:
            try:
                lat = float(d.get("y"))
                lng = float(d.get("x"))
            except:
                continue

            name = d.get("place_name") or d.get("address_name") or d.get("road_address_name") or query
            results.append({"name": name, "lat": lat, "lng": lng})
            if len(results) >= limit:
                break

        meta = js.get("meta", {})
        if meta.get("is_end", True):
            break

        page += 1

    return results


def _kakao_address_search(query: str, limit: int = 30) -> List[Dict[str, Any]]:
    key = _get_kakao_rest_key()
    if not key:
        raise RuntimeError("KAKAO_REST_API_KEY is not set")

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    size = min(30, max(1, int(limit)))
    page = 1
    results: List[Dict[str, Any]] = []

    while len(results) < limit and page <= 45:
        params = {"query": query, "page": page, "size": size}
        r = requests.get(url, params=params, headers=_kakao_headers(), timeout=10)

        if r.status_code != 200:
            raise RuntimeError(f"Kakao address search failed: {r.status_code} {r.text}")

        js = r.json()
        docs = js.get("documents", [])

        for d in docs:
            # ✅ address.json은 문서에 x/y가 top-level로 오는 경우가 많아서 우선 사용
            x = d.get("x")
            y = d.get("y")

            # fallback
            addr = d.get("address") or {}
            if (x is None) or (y is None):
                x = x or addr.get("x")
                y = y or addr.get("y")

            try:
                lat = float(y)
                lng = float(x)
            except:
                continue

            name = d.get("address_name") or query
            results.append({"name": name, "lat": lat, "lng": lng})
            if len(results) >= limit:
                break

        meta = js.get("meta", {})
        if meta.get("is_end", True):
            break

        page += 1

    return results


def _dedup_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = (round(float(it["lat"]), 6), round(float(it["lng"]), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# -----------------------------
# Model load
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("CVRP_MODEL", str(DEFAULT_MODEL_PATH))

if not Path(MODEL_PATH).exists():
    print(f"[WARN] Model not found: {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=device)
n_customers_trained = ckpt.get("n_customers", None)

model = CVRPModel(embed_dim=128, n_heads=8, n_layers=3).to(device)
model.load_state_dict(ckpt["model"])
model.eval()


@app.get("/")
def home():
    if not WEB_INDEX.exists():
        return {"error": f"index.html not found at {WEB_INDEX}"}
    return FileResponse(WEB_INDEX, media_type="text/html")


@app.get("/health")
def health():
    key = _get_kakao_rest_key()
    return {
        "ok": True,
        "device": device,
        "trained_n_customers": n_customers_trained,
        "split_mode": "capacity_only",
        "commit": os.environ.get("RENDER_GIT_COMMIT"),
        "geocoder": "kakao" if key else "nominatim_fallback",
        "kakao_key_loaded": bool(key),
        "kakao_key_len": len(key),
    }


@app.get("/geocode")
def geocode(q: str, limit: int = 30):
    q = (q or "").strip()
    if not q:
        return {"results": []}

    key = _get_kakao_rest_key()

    # ✅ Kakao 우선
    if key:
        try:
            results = _kakao_keyword_search(q, limit=limit)

            if len(results) < max(5, min(limit, 10)):
                more = _kakao_address_search(q, limit=limit)
                results = _dedup_results(results + more)

            results = results[: max(1, int(limit))]
            return {"results": results}

        except Exception as e:
            # ✅ 카카오가 막히면 이유를 그대로 반환 (디버그용)
            return {"error": str(e), "results": []}

    # ✅ fallback: nominatim
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": q,
        "format": "json",
        "limit": str(limit),
        "countrycodes": "kr",
        "addressdetails": "1",
    }
    headers = {
        "User-Agent": "cvrp-demo/1.0 (educational project)",
        "Accept-Language": "ko",
    }
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data:
        results.append({
            "name": item.get("display_name", ""),
            "lat": float(item["lat"]),
            "lng": float(item["lon"]),
        })
    return {"results": results}


@app.post("/route")
def route(req: RouteRequest):
    if len(req.coords) < 2:
        return {"error": "need at least 2 coords"}

    coord_str = ";".join([f"{lng},{lat}" for lat, lng in req.coords])
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}"
    params = {"overview": "full", "geometries": "geojson"}

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()

    if "routes" not in js or len(js["routes"]) == 0:
        return {"error": "no route returned", "raw": js}

    route0 = js["routes"][0]
    coords_lonlat = route0["geometry"]["coordinates"]
    polyline = [[lat, lon] for lon, lat in coords_lonlat]

    return {"distance_m": route0["distance"], "duration_s": route0["duration"], "polyline": polyline}


@app.post("/solve")
def solve(req: SolveRequest):
    if not Path(MODEL_PATH).exists():
        return {"error": f"model not found: {MODEL_PATH}"}

    coords = torch.tensor(req.coords, dtype=torch.float32)
    demands = torch.tensor(req.demands, dtype=torch.float32)
    cap = float(req.capacity)

    if coords.ndim != 2 or coords.size(1) != 2:
        return {"error": "coords must be (N,2)"}
    if demands.ndim != 1 or demands.size(0) != coords.size(0):
        return {"error": "demands length must equal coords length"}
    if float(demands[0].item()) != 0.0:
        return {"error": "demands[0] must be 0 (depot)"}
    if req.decode not in ["greedy", "sampling"]:
        return {"error": "decode must be 'greedy' or 'sampling'"}
    if not (cap > 0):
        return {"error": "capacity must be > 0"}

    for i in range(1, demands.size(0)):
        if float(demands[i].item()) > cap:
            return {"error": f"Customer {i} demand({float(demands[i].item())}) > capacity({cap})"}

    coords_n = _normalize_uniform(coords)

    coords_b = coords_n.unsqueeze(0).to(device)
    demands_b = demands.unsqueeze(0).to(device)
    cap_b = torch.tensor([cap], dtype=torch.float32).to(device)

    with torch.no_grad():
        raw_path, _ = model(coords_b, demands_b, cap_b, decode_type=req.decode)
        raw_path = raw_path[0].tolist()
        path = _clean_path(raw_path, n_nodes=coords.size(0))

        dist = route_length(
            coords_b,
            torch.tensor(path, device=device).unsqueeze(0)
        ).item()

    try:
        trips, visit_order = split_trips_capacity_only(
            path=path,
            demands=demands.tolist(),
            capacity=cap,
        )
    except Exception as e:
        return {"error": str(e)}

    trip_loads = [_trip_load(t, demands.tolist()) for t in trips]

    return {
        "path": path,
        "visit_order": visit_order,
        "trips": trips,
        "trip_loads": trip_loads,
        "distance_norm": dist,
    }
