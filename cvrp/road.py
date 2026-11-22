"""
OSRM 도로 거리 행렬 (평가용)

- 공개 OSRM 서버의 table API로 지점 간 도로 거리(m)·시간(s) 행렬을 받는다.
- 공개 서버 사용 정책(초당 1회 이하)을 지키도록 요청 간격을 둔다.
- 평가셋 단위로 디스크에 캐시해(results/road_cache/*.npz) 다시 돌릴 때 OSRM을 부르지 않는다.
  캐시는 저장소에 커밋해 누구나 같은 숫자를 재현할 수 있게 한다.
- 도로 거리는 일방통행 등으로 비대칭(dist[i][j] != dist[j][i])일 수 있다.
"""
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

OSRM_URL = "https://router.project-osrm.org"
MIN_INTERVAL_S = 1.1  # 공개 서버 정책: 초당 1회 이하
USER_AGENT = "cvrp-road-demo-eval/1.0"

_last_call = 0.0


def fetch_osrm_table(
    coords_latlng: Sequence[Sequence[float]],
    base_url: str = OSRM_URL,
    timeout: float = 30.0,
    retries: int = 3,
    min_interval: float = MIN_INTERVAL_S,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [lat, lng] 목록 → (거리 m, 시간 s) 행렬. 실패하면 재시도 후 예외.
    평가 스크립트는 기본값(초당 1회, 3회 시도)을, 데모 서버는 사용자 요청마다
    한 번만 부르므로 retries=1, min_interval=0 과 짧은 timeout을 쓴다.
    """
    global _last_call
    coord_str = ";".join(f"{lng:.6f},{lat:.6f}" for lat, lng in coords_latlng)
    url = f"{base_url}/table/v1/driving/{coord_str}?annotations=distance,duration"

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        wait = min_interval - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                js = json.loads(r.read())
            if js.get("code") != "Ok":
                raise RuntimeError(f"OSRM error: {js.get('code')} {js.get('message')}")
            dist = np.array(js["distances"], dtype=np.float64)
            dur = np.array(js["durations"], dtype=np.float64)
            if np.isnan(dist).any() or np.isnan(dur).any():
                raise RuntimeError("OSRM returned null entries (unroutable point)")
            return dist, dur
        except Exception as e:  # 네트워크·429 등 → 잠시 쉬고 재시도
            last_err = e
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"OSRM table failed after {retries} tries: {last_err}")


class RoadCache:
    """평가셋 하나의 도로 행렬 캐시. 인스턴스 번호 → (dist_m, dur_s)."""

    def __init__(self, path: Path, fetch: Callable = fetch_osrm_table):
        self.path = Path(path)
        self.fetch = fetch
        self.data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.dirty = False
        if self.path.exists():
            with np.load(self.path) as z:
                for key in z.files:
                    kind, idx = key.split("_")
                    d = self.data.setdefault(int(idx), [None, None])
                    d[0 if kind == "dist" else 1] = z[key].astype(np.float64)
            self.data = {k: (v[0], v[1]) for k, v in self.data.items()}

    @staticmethod
    def _as_stored(a: np.ndarray) -> np.ndarray:
        # 캐시는 float32로 저장하므로, 처음 받은 값도 같은 정밀도로 맞춰 재실행 결과를 일치시킨다
        return np.asarray(a, dtype=np.float32).astype(np.float64)

    def get(self, idx: int, coords_latlng: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        hit = self.data.get(idx)
        if hit is not None and hit[0].shape[0] == len(coords_latlng):
            return hit
        dist, dur = self.fetch(coords_latlng)
        dist, dur = self._as_stored(dist), self._as_stored(dur)
        self.data[idx] = (dist, dur)
        self.dirty = True
        return dist, dur

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for idx, (dist, dur) in self.data.items():
            arrays[f"dist_{idx}"] = dist.astype(np.float32)
            arrays[f"dur_{idx}"] = dur.astype(np.float32)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")  # np.savez는 .npz를 붙인다
        np.savez_compressed(tmp, **arrays)
        tmp.replace(self.path)
        self.dirty = False
