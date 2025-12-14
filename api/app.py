# api/app.py
import os
from pathlib import Path
from typing import List, Optional, Tuple

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
    """
    모델이 내는 path가 이런 형태일 수 있음:
    - 끝에 0 패딩이 길게 붙음
    - 연속 0이 많음
    - 혹시 범위를 벗어난 인덱스가 섞임

    ✅ 디버그/로그 가독성을 위해 정리:
    - int로 캐스팅
    - [0..n-1] 범위 밖 제거
    - 연속 0 압축
    - 마지막 의미 있는 방문 뒤로 trailing 0 제거 후, 끝은 0으로 보정
    """
    # 1) int 캐스팅 + 범위 필터
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

    # 2) 연속 0 압축
    compact = [p[0]]
    for v in p[1:]:
        if v == 0 and compact[-1] == 0:
            continue
        compact.append(v)

    # 3) trailing 0 제거(마지막 고객 뒤까지만 남김)
    last_nonzero = -1
    for i in range(len(compact) - 1, -1, -1):
        if compact[i] != 0:
            last_nonzero = i
            break

    if last_nonzero == -1:
        # 전부 0이면
        return [0, 0]

    compact = compact[: last_nonzero + 1]

    # 4) 시작/끝 depot 보정
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
    """
    ✅ '용량이 부족할 때만' depot(0) 복귀하여 회차를 나누는 규칙

    - path 안의 0 때문에 회차가 쪼개지는 걸 무시하고,
    - 고객 방문 순서(visit_order)만 뽑아서
    - capacity 초과 시에만 trips를 분할한다.
    """
    n = len(demands)

    # 1) 고객 방문 순서 추출 (0 제거 + 중복 제거)
    order: List[int] = []
    seen = {0}
    for v in path:
        if v != 0 and (v not in seen) and (1 <= v < n):
            order.append(v)
            seen.add(v)

    # (안전장치) 혹시 누락 고객이 있으면 뒤에 붙임
    for i in range(1, n):
        if i not in seen:
            order.append(i)

    # 2) 용량 기준으로만 회차 분할
    trips: List[List[int]] = []
    cur = [0]
    load = 0.0

    for c in order:
        d = float(demands[c])
        if d > capacity:
            raise ValueError(f"Customer {c} demand({d}) > capacity({capacity})")

        # 다음 고객을 싣고 가면 초과되는 순간에만 회차 종료
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
    s = 0.0
    for idx in trip:
        if idx != 0:
            s += float(demands[idx])
    return s


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
    return {
        "ok": True,
        "device": device,
        "trained_n_customers": n_customers_trained,
        "split_mode": "capacity_only",
        "commit": os.environ.get("RENDER_GIT_COMMIT"),
    }


@app.get("/geocode")
def geocode(q: str, limit: int = 5):
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

    # OSRM: lon,lat
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

    return {
        "distance_m": route0["distance"],
        "duration_s": route0["duration"],
        "polyline": polyline,
    }


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

    # 개별 고객 demand > capacity면 불가능
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

        # ✅ 디버그/가독성용 path 정리
        path = _clean_path(raw_path, n_nodes=coords.size(0))

        # distance는 "정리된 path" 기준으로 계산(보는 값이랑 일치)
        dist = route_length(
            coords_b,
            torch.tensor(path, device=device).unsqueeze(0)
        ).item()

    # ✅ trips는 "용량 기준"으로만 만든다(모델이 0 남발해도 무시)
    try:
        trips, visit_order = split_trips_capacity_only(
            path=path,
            demands=demands.tolist(),
            capacity=cap,
        )
    except Exception as e:
        return {"error": str(e)}

    trip_loads = [ _trip_load(t, demands.tolist()) for t in trips ]

    return {
        "path": path,                    # 정리된 path(디버그용)
        "visit_order": visit_order,      # 고객 방문 순서(0 제외)
        "trips": trips,                  # ✅ capacity 기준 회차
        "trip_loads": trip_loads,        # (프론트 표에 쓰기 좋음)
        "distance_norm": dist,
    }
