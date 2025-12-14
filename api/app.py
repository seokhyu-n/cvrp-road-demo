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

# ✅ Render Environment Variables 에 넣을 값 이름
#   KEY: KAKAO_REST_API_KEY
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "").strip()


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
    ✅ 디버그/로그 가독성을 위해 정리:
    - int로 캐스팅
    - [0..n-1] 범위 밖 제거
    - 연속 0 압축
    - 마지막 의미 있는 방문 뒤로 trailing 0 제거 후, 끝은 0으로 보정
    """
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

    # 연속 0 압축
    compact = [p[0]]
    for v in p[1:]:
        if v == 0 and compact[-1] == 0:
            continue
        compact.append(v)

    # trailing 0 제거(마지막 고객 뒤까지만 남김)
    last_nonzero = -1
    for i in range(len(compact) - 1, -1, -1):
        if compact[i] != 0:
            last_nonzero = i
            break

    if last_nonzero == -1:
        return [0, 0]

    compact = compact[: last_nonzero + 1]

    # 시작/끝 depot 보정
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
    """
    n = len(demands)

    # 고객 방문 순서 추출 (0 제거 + 중복 제거)
    order: List[int] = []
    seen = {0}
    for v in path:
        if v != 0 and (v not in seen) and (1 <= v < n):
            order.append(v)
            seen.add(v)

    # 누락 고객 있으면 뒤에 붙임(안전장치)
    for i in range(1, n):
        if i not in seen:
            order.append(i)

    # 용량 기준으로만 회차 분할
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
    s = 0.0
    for idx in trip:
        if idx != 0:
            s += float(demands[idx])
    return s


# -----------------------------
# ✅ Kakao geocode helpers
# -----------------------------
def _kakao_headers() -> Dict[str, str]:
    # 카카오 로컬 API 인증 헤더: Authorization: KakaoAK ${REST_API_KEY} :contentReference[oaicite:2]{index=2}
    return {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}


def _kakao_keyword_search(query: str, limit: int = 30) -> List[Dict[str, Any]]:
    """
    카카오 '키워드로 장소 검색'을 page 반복 호출해서 limit 개수까지 합친다.
    - size 최대 15 :contentReference[oaicite:3]{index=3}
    - page 최대 45 :contentReference[oaicite:4]{index=4}
    """
    if not KAKAO_REST_API_KEY:
        raise RuntimeError("KAKAO_REST_API_KEY is not set")

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    size = min(15, max(1, int(limit)))  # 카카오 size 최대 15 :contentReference[oaicite:5]{index=5}
    page = 1
    results: List[Dict[str, Any]] = []

    while len(results) < limit and page <= 45:  # page 최대 45 :contentReference[oaicite:6]{index=6}
        params = {"query": query, "page": page, "size": size}
        r = requests.get(url, params=params, headers=_kakao_headers(), timeout=10)
        r.raise_for_status()
        js = r.json()

        docs = js.get("documents", [])
        for d in docs:
            # kakao: x=longitude, y=latitude
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
    """
    카카오 '주소로 장소 검색' (fallback 용)
    - size 최대 30, page 최대 45 같은 형태로 제공되는 섹션이 있음(문서 기준). :contentReference[oaicite:7]{index=7}
    """
    if not KAKAO_REST_API_KEY:
        raise RuntimeError("KAKAO_REST_API_KEY is not set")

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    size = min(30, max(1, int(limit)))
    page = 1
    results: List[Dict[str, Any]] = []

    while len(results) < limit and page <= 45:
        params = {"query": query, "page": page, "size": size}
        r = requests.get(url, params=params, headers=_kakao_headers(), timeout=10)
        r.raise_for_status()
        js = r.json()

        docs = js.get("documents", [])
        for d in docs:
            addr = d.get("address") or {}
            try:
                lat = float(addr.get("y"))
                lng = float(addr.get("x"))
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
    return {
        "ok": True,
        "device": device,
        "trained_n_customers": n_customers_trained,
        "split_mode": "capacity_only",
        "commit": os.environ.get("RENDER_GIT_COMMIT"),
        "geocoder": "kakao" if KAKAO_REST_API_KEY else "nominatim_fallback",
    }


@app.get("/geocode")
def geocode(q: str, limit: int = 30):
    """
    ✅ Kakao 로컬 검색 기반
    - 키워드 검색 우선
    - 결과가 너무 적으면 주소 검색도 추가로 합쳐서 반환
    """
    q = (q or "").strip()
    if not q:
        return {"results": []}

    # 카카오 키가 없으면 (실수로 env를 안 넣은 경우) 기존 nominatim fallback
    if not KAKAO_REST_API_KEY:
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

    # ✅ Kakao keyword search
    results = _kakao_keyword_search(q, limit=limit)

    # 부족하면 주소검색 결과도 붙여보기(아파트/동/지번 같은 케이스 보완)
    if len(results) < max(5, min(limit, 10)):
        more = _kakao_address_search(q, limit=limit)
        results = _dedup_results(results + more)

    # limit 컷
    results = results[: max(1, int(limit))]
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

    trip_loads = [_trip_load(t, demands.tolist()) for t in trips]

    return {
        "path": path,
        "visit_order": visit_order,
        "trips": trips,
        "trip_loads": trip_loads,
        "distance_norm": dist,
    }
