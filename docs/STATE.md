# STATE (2026-11-22)

## 지금 상태
- main = ac03121 (PR #2~#4). Render 배포 완료, 라이브 고객 추가·Solve 정상.
- 브랜치 `feat/road-eval` (PR #5): OSRM 도로 거리 기준 평가(`--metric road`), Run-05/06, road_cache, 데모 /solve 도로 거리 선택(E) 기본 적용.
- 도로 거리 기준(Run-05/06): 직선 선택 D gap +17.8% / +21.1% → 도로 선택 E +11.2% / +13.0%.
- 로컬 확인: /solve road 약 1.0초(OSRM 호출 포함), straight 0.07초. 예시 20명에서 실제 도로거리 −5.1%.

## 다음 할 일
1. PR #5 머지 후 라이브에서 도로 거리 모드 응답 시간·동작 확인.
2. v1 → v2 비포·애프터 정리: `v1.0-school` 태그(d044b0c)·Release, README 상단 비교표, v1 내용 보존, 새 스크린샷.
3. 소소한 정리: 에러를 HTTP 400으로, CORS 정리.
4. 재학습(Colab GPU): 효과는 도로 거리 기준 gap으로 측정.

## 막힌 점
- 없음.

## 실행 방법
- 테스트: `.\.venv\Scripts\python.exe -m pytest -q` (33개, OSRM은 테스트에서 가짜로 대체)
- 평가(직선): `.\.venv\Scripts\python.exe -m cvrp.eval_pipeline --instances 200 --customers 20 --samples 32 --note "..."`
- 평가(도로): 위 명령에 `--metric road` (캐시 있으면 OSRM 호출 없음)
- 기록 보기: `.\.venv\Scripts\python.exe -m cvrp.runlog --eval-set road`
- 라이브 확인: https://cvrp-road-demo.onrender.com/health
