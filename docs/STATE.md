# STATE (2026-11-22)

## 지금 상태
- main = 3b48198 (PR #2 추론 개선, PR #3 OR-Tools 비교·스레드 고정·docs). Render 배포 완료, `torch_threads: 1`.
- 현재 기본값 D(정규화 + best-of-32 + 2-opt): NN 대비 n20 +13.2% / n50 +14.3%, OR-Tools 대비 gap +10.9% / +10.1% (Run-03/04).
- 라이브 /solve (네트워크 포함, 샘플 32): 고객 8명 0.4초(스레드 고정 전 7~9초), 20명 0.8초, 50명 약 2.2초.

## 다음 할 일
1. 도로 거리 기준 평가: OSRM table API로 거리 행렬을 받아 캐시하고, 2-opt·평가를 도로 거리로.
2. 재학습(Colab GPU): step 증가, 고객 수 혼합(20~50), 고정 baseline(rollout baseline). OR-Tools gap으로 효과 측정.
3. 소소한 정리: 에러를 HTTP 400으로, CORS 정리.

## 막힌 점
- 브라우저에서 장소 검색 → Solve 클릭 확인은 아직 안 함 (로컬은 Kakao 키 없음, 라이브 데모에서 가능).

## 실행 방법
- 테스트: `.\.venv\Scripts\python.exe -m pytest -q` (24개)
- 평가: `.\.venv\Scripts\python.exe -m cvrp.eval_pipeline --instances 200 --customers 20 --samples 32 --note "..."`
- 기록 보기: `.\.venv\Scripts\python.exe -m cvrp.runlog --method D`
- 라이브 확인: https://cvrp-road-demo.onrender.com/health
