# api/app.py
import os
from pathlib import Path
from typing import List, Optional

import requests
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cvrp.model import CVRPModel
from cvrp.data import split_trips, route_length

app = FastAPI(title="CVRP Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 같은 origin으로 쓸 거면 사실 필요 없지만 둬도 됨
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
    mins = coords_tensor.min(dim=0).values
    maxs = coords_tensor.max(dim=0).values
    span = max((maxs - mins).max().item(), 1e-9)
    return (coords_tensor - mins) / span

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.environ.get("CVRP_MODEL", str(DEFAULT_MODEL_PATH))
if not Path(MODEL_PATH).exists():
    # 모델이 없으면 서버는 뜨되 solve에서 에러 안내하게 할 수도 있음
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
    return {"ok": True, "device": device, "trained_n_customers": n_customers_trained}

@app.get("/geocode")
def geocode(q: str, limit: int = 5):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": q,
        "format": "json",
        "limit": str(limit),
        "countrycodes": "kr",
        "addressdetails": "1"
    }
    headers = {
        "User-Agent": "cvrp-demo/1.0 (educational project)",
        "Accept-Language": "ko"
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
    cap = torch.tensor([req.capacity], dtype=torch.float32)

    if coords.ndim != 2 or coords.size(1) != 2:
        return {"error": "coords must be (N,2)"}
    if demands.ndim != 1 or demands.size(0) != coords.size(0):
        return {"error": "demands length must equal coords length"}
    if float(demands[0].item()) != 0.0:
        return {"error": "demands[0] must be 0 (depot)"}
    if req.decode not in ["greedy", "sampling"]:
        return {"error": "decode must be 'greedy' or 'sampling'"}

    coords_n = _normalize_uniform(coords)

    coords_b = coords_n.unsqueeze(0).to(device)
    demands_b = demands.unsqueeze(0).to(device)
    cap_b = cap.to(device)

    with torch.no_grad():
        path, _ = model(coords_b, demands_b, cap_b, decode_type=req.decode)
        path = path[0].tolist()
        dist = route_length(coords_b, torch.tensor(path, device=device).unsqueeze(0)).item()

    trips = split_trips(path)
    return {"path": path, "trips": trips, "distance_norm": dist}
