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
    allow_origins=["*"],  # 같은 origin으로만 쓸 거면 없어도 되지만 둬도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 프로젝트 루트 경로 (api/의 상위 폴더)
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
    """Uniform normalization (translation + uniform scale)."""
    mins = coords_tensor.min(dim=0).values
    maxs = coords_tensor.max(dim=0).values
    span = max((maxs - mins).max().item(), 1e-9)
    return (coords_tensor - mins) / span


def split_trips_capacity_only(
    path: List[int],
    demands: List[float],
    capacity: float
) -> Tuple[List[List[int]], List[int]]:
    """
    ✅ '중간 창고 복귀 금지(용량 부족할 때만 복귀)' 규칙 구현.

    - 모델이 낸 path에는 0(depot)이 여러 번 섞여 있을 수 있음.
    - 우리는 '고객 방문 순서'만 추출(0 제거)하고,
    - 용량(capacity)을 넘길 때만 depot(0)로 돌아가도록 회차를 분할한다.

    반환:
      trips: [[0, a, b, 0], [0, c, 0], ...]
      order: [a, b, c, ...]  (0 제외 고객 방문 순서)
    """
    n = len(demands)

    # 1) 고객 방문 순서 추출(0 제거 + 중복 제거)
    order: List[int] = []
    seen = set([0])
    for v in path:
        if v != 0 and v not in seen and 1 <= v < n:
            order.append(v)
            seen.add(v)

    # (안전장치) 혹시 모델이 누락한 고객이 있으면 뒤에 붙임
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

        # 다음 고객을 실으면 초과되는 순간에만 회차 종료
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


device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.environ.get("CVRP_MODEL", str(DEFAULT_MODEL_PATH))
if not Path(MODEL_PATH).exists():
    print(f"[WARN] Model not found: {MODEL_PATH}")

# 모델은 서버 시작 시 로딩(현재 구조 유지)
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
    return {"ok": True, "device": device, "trained_n_customers": n_customers_trained}


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

    # OSRM expects lon,lat
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
    cap = torch.tensor([req.capacity], dtype=torch.float32)

    if coords.ndim != 2 or coords.size(1) != 2:
        return {"error": "coords must be (N,2)"}
    if demands.ndim != 1 or demands.size(0) != coords.size(0):
        return {"error": "demands length must equal coords length"}
    if float(demands[0].item()) != 0.0:
        return {"error": "demands[0] must be 0 (depot)"}
    if req.decode not in ["greedy", "sampling"]:
        return {"error": "decode must be 'greedy' or 'sampling'"}
    if not (req.capacity > 0):
        return {"error": "capacity must be > 0"}

    coords_n = _normalize_uniform(coords)

    coords_b = coords_n.unsqueeze(0).to(device)
    demands_b = demands.unsqueeze(0).to(device)
    cap_b = cap.to(device)

    with torch.no_grad():
        path, _ = model(coords_b, demands_b, cap_b, decode_type=req.decode)
        path = path[0].tolist()

        dist = route_length(
            coords_b,
            torch.tensor(path, device=device).unsqueeze(0)
        ).item()

    # ✅ 여기서 핵심 변경: '용량 부족할 때만' 회차 분할
    try:
        trips, visit_order = split_trips_capacity_only(
            path=path,
            demands=demands.tolist(),
            capacity=float(req.capacity),
        )
    except Exception as e:
        return {"error": str(e)}

    return {
        "path": path,                 # 모델 원본 출력(디버그용)
        "visit_order": visit_order,   # 고객 방문 순서(0 제외)
        "trips": trips,               # ✅ 우리가 원하는 규칙으로 만든 회차
        "distance_norm": dist,
    }
