# STATE (2026-11-24)

## 지금 상태
- v2 + 웹 화면 개편까지 main에 반영, Render 라이브도 같은 화면. README의 v2 실행 화면도 새 화면으로 교체.
- 최근 PR: #9 오류 메시지를 화면 안에 표시 · #10 웹 화면 가독성 개편(한글 버튼, 1·2·3 단계 카드) · README 화면 갱신.
- 릴리스: `v1.0-school`(d044b0c, 학교 원본) · `v2.0`(a0614ed, Latest). README 상단에 v1→v2 비교표.
- 성능(도로 거리, Run-05·06): v2 기본값 E는 OR-Tools보다 11.2% / 13.0% 김, NN보다 13.3% / 12.2% 짧음 (20명 / 50명).
- 라이브 응답: 8명 약 1.2초, 20명 1.7~2.0초, 50명 약 3.5초 (도로 거리 모드, OSRM 조회 약 1초 포함).

## 다음 할 일 (추천 순서, 상세는 docs/NEXT_STEPS.md)
1. GitHub Actions CI (pytest 자동 실행 + README 배지) — 30분.
2. 후처리 강화(학습 없이): 최적 분할(Split DP) + 회차 간 지역 탐색(relocate/swap/or-opt). Run-07~로 기록.
3. POMO 방식 재학습(Colab GPU): 고객 수 20~50 혼합. 결과는 v3로 비포·애프터.
4. 선택: README에 "왜 딥러닝인가"(전공 과목이 딥러닝) 한두 줄, GitHub About/토픽.

## 막힌 점
- 없음.

## 실행 방법
- 새 PC: `git clone` → `python -m venv .venv` → CPU torch 설치 → `-r requirements-dev.txt` (CLAUDE.md). 기존 클론은 `git pull`. Python 3.11/3.12 모두 테스트 통과.
- 테스트: `.\.venv\Scripts\python.exe -m pytest -q` (36개, OSRM은 테스트에서 가짜로 대체)
- 서버: `.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8765` (localhost 대신 127.0.0.1: Windows IPv6 지연). Claude 앱에서는 `.claude/launch.json`의 `cvrp-server`로도 실행.
- 평가: `.\.venv\Scripts\python.exe -m cvrp.eval_pipeline --instances 200 --customers 20 --samples 32 [--metric road] --note "..."`
- 기록 보기: `.\.venv\Scripts\python.exe -m cvrp.runlog --eval-set road`
- 라이브 확인: https://cvrp-road-demo.onrender.com/health (`commit`이 main과 같은지). 무료 플랜이라 처음 접속 시 깨어나는 데 1분 정도 걸림.
