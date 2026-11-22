# STATE (2026-11-22)

## 지금 상태
- main = ac03121 (PR #2~#4). Render 배포 완료, 라이브 고객 추가·Solve 정상.
- 브랜치 `feat/road-eval`: OSRM 도로 거리 기준 평가(`--metric road`), Run-05/06 기록, road_cache 커밋.
- 도로 거리 기준(Run-05/06): 현재 데모 D gap +17.8% / +21.1%, 도로 거리로 선택·2-opt하는 E gap +11.2% / +13.0%.
  → 데모에 E(도로 거리 선택)를 넣으면 실제 도로 기준으로 약 6~8% 더 짧아질 것으로 보임.

## 다음 할 일
1. (결정 필요) 데모 /solve에 도로 거리 선택(E) 적용: OSRM table 1회 호출, 실패 시 직선거리로 대체.
2. v1 → v2 비포·애프터 정리: `v1.0-school` 태그(d044b0c)·Release, README 상단 비교표, v1 내용 보존, 새 스크린샷.
3. 소소한 정리: 에러를 HTTP 400으로, CORS 정리.
4. 재학습(Colab GPU): 효과는 도로 거리 기준 gap으로 측정.

## 막힌 점
- 없음.

## 실행 방법
- 테스트: `.\.venv\Scripts\python.exe -m pytest -q` (29개)
- 평가(직선): `.\.venv\Scripts\python.exe -m cvrp.eval_pipeline --instances 200 --customers 20 --samples 32 --note "..."`
- 평가(도로): 위 명령에 `--metric road` (캐시 있으면 OSRM 호출 없음)
- 기록 보기: `.\.venv\Scripts\python.exe -m cvrp.runlog --eval-set road`
- 라이브 확인: https://cvrp-road-demo.onrender.com/health
