# api/app.py
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cvrp.model import CVRPModel
from cvrp.solver import NORMALIZE_MODES, solve_instance

MAX_SAMPLES = 128
# 공개 데모라 요청 하나가 서버를 오래 붙잡지 않도록 상한을 둔다
MAX_NODES = 101                    # 창고 1 + 고객 100
MAX_SAMPLE_NODES = MAX_SAMPLES * 51  # n_samples × 지점 수 상한 (고객 50명이면 샘플 128까지)

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
    n_samples: int = 32         # best-of-N 샘플 수 (0이면 greedy만)
    two_opt: bool = True        # 회차별 2-opt 후처리
    normalize: str = "depot_center"  # "depot_center" | "minmax"(기존 방식)
    seed: int = 0
    decode: Optional[str] = None     # (하위호환) "greedy"면 n_samples=0


class RouteRequest(BaseModel):
    coords: List[List[float]]   # [lat,lng] visiting order


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

model: Optional[CVRPModel] = None
n_customers_trained = None

if Path(MODEL_PATH).exists():
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    n_customers_trained = ckpt.get("n_customers", None)
    model = CVRPModel(embed_dim=128, n_heads=8, n_layers=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
else:
    print(f"[WARN] Model not found: {MODEL_PATH}")


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
    if model is None:
        return {"error": f"model not found: {MODEL_PATH}"}

    n = len(req.coords)
    cap = float(req.capacity)

    if n < 2 or any(len(c) != 2 for c in req.coords):
        return {"error": "coords must be (N,2) with N >= 2"}
    if len(req.demands) != n:
        return {"error": "demands length must equal coords length"}
    if float(req.demands[0]) != 0.0:
        return {"error": "demands[0] must be 0 (depot)"}
    if req.decode not in (None, "greedy", "sampling"):
        return {"error": "decode must be 'greedy' or 'sampling'"}
    if req.normalize not in NORMALIZE_MODES:
        return {"error": f"normalize must be one of {list(NORMALIZE_MODES)}"}
    if not (cap > 0):
        return {"error": "capacity must be > 0"}
    if n > MAX_NODES:
        return {"error": f"too many points: {n} (max {MAX_NODES}, depot included)"}
    if not (0 <= req.n_samples <= MAX_SAMPLES):
        return {"error": f"n_samples must be in [0, {MAX_SAMPLES}]"}
    if req.n_samples * n > MAX_SAMPLE_NODES:
        return {"error": f"n_samples × points too large (max {MAX_SAMPLE_NODES}); lower n_samples"}

    for i in range(1, n):
        if float(req.demands[i]) > cap:
            return {"error": f"Customer {i} demand({float(req.demands[i])}) > capacity({cap})"}

    n_samples = 0 if req.decode == "greedy" else req.n_samples

    try:
        result = solve_instance(
            model,
            req.coords,
            req.demands,
            cap,
            n_samples=n_samples,
            use_two_opt=req.two_opt,
            normalize=req.normalize,
            seed=req.seed,
            device=device,
        )
    except Exception as e:
        return {"error": str(e)}

    return {
        "visit_order": result["visit_order"],
        "trips": result["trips"],
        "trip_loads": result["trip_loads"],
        "n_candidates": result["n_candidates"],
        "stats": result["stats"],
        "settings": {
            "n_samples": n_samples,
            "two_opt": req.two_opt,
            "normalize": req.normalize,
        },
    }
