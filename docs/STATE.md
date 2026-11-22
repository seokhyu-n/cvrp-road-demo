# STATE (2026-11-22)

## 지금 상태
- main = 003a88e (PR #2~#5). 라이브는 도로 거리 선택(E) 기본, 응답 8명 약 1.2초 / 20명 1.7~2.0초 / 50명 3.5초.
- 태그 `v1.0-school`(d044b0c) + GitHub Release 생성: 학교 프로젝트 원본 보존.
- 브랜치 `feat/v2-showcase`: README v1→v2 비포·애프터 재구성, v2 스크린샷(docs/images/demo_v2.png),
  HTTP 400/502/503 상태 코드, CORS 기본 끔(CORS_ORIGINS로 지정), 체크박스 레이아웃 수정, 디버그 JSON 접기.

## 다음 할 일
1. PR 머지 후 라이브 확인, `v2.0` 태그 + Release (v1 대비 개선 요약).
2. 재학습(Colab GPU): step 증가, 고객 수 혼합(20~50), rollout baseline. 도로 거리 기준 gap으로 측정.

## 막힌 점
- 없음.

## 실행 방법
- 테스트: `.\.venv\Scripts\python.exe -m pytest -q` (36개, OSRM은 테스트에서 가짜로 대체)
- 평가(직선): `.\.venv\Scripts\python.exe -m cvrp.eval_pipeline --instances 200 --customers 20 --samples 32 --note "..."`
- 평가(도로): 위 명령에 `--metric road` (캐시 있으면 OSRM 호출 없음)
- 기록 보기: `.\.venv\Scripts\python.exe -m cvrp.runlog --eval-set road`
- 라이브 확인: https://cvrp-road-demo.onrender.com/health
