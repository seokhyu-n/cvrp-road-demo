# HISTORY

## 2026-11-22 — 라이브 응답 시간 재측정 (PR #3 머지 후, 3b48198)
- 한 일: Render 배포 후 /solve를 샘플 수별로 3회씩 측정(네트워크 포함).
- 결과: 고객 8명·샘플 32 7~9초 → 0.4초(약 20배). 20명 0.8초, 50명 약 2.2초. greedy만은 약 0.25초.
- 결정: 스레드 고정이 원인 해결로 확인됨. 웹 기본 샘플 수 32 유지.
- 다음 할 일: 도로 거리(OSRM table) 기준 평가, 재학습.

## 2026-11-22 — OR-Tools 참조해 + 작업 상태 문서 + Render 속도
- 한 일: 평가에 OR-Tools(GLS, 1s/2s) 참조해와 gap 지표 추가, runs.csv에 gap_ort_pct 컬럼(헤더 자동 이전). Run-03(n20)·Run-04(n50) 기록. CLAUDE.md와 docs/SPEC·STATE·HISTORY 작성.
- 결과: D의 OR-Tools 대비 gap +10.9% / +10.1%, 기존 데모(A)는 +27.3% / +32.6%. D는 OR-Tools보다 15~20배 빠름.
- 발견: 라이브 /solve가 7~9초(greedy만 1.2초) → PyTorch 스레드 경합 의심, TORCH_NUM_THREADS 기본 1로 고정. /health에서 키 길이 대신 torch_threads 표시.
- 버그 수정: runs.csv 자체 변경 때문에 git_commit이 -dirty로 찍히던 문제 → 기록 파일은 dirty 판정에서 제외, Run-04 표기를 b44c587로 정정.
- 결정: OR-Tools는 평가 전용이라 requirements-dev.txt로 분리(Render 배포 의존성에 넣지 않음).
- 다음 할 일: 머지 후 라이브 속도 재측정, 도로 거리(OSRM table) 기준 평가, 재학습.

## 2026-11-22 — 추론 파이프라인 개선 (PR #2, squash 34f719f)
- 한 일: cvrp/solver.py로 추론 분리. 경도 cos(위도) 보정 + 창고 중앙 정규화, best-of-32 샘플링, 회차별 2-opt. /solve 옵션·요청 상한, 웹 UI 옵션, 평가 스크립트와 results/runs.csv Run 기록.
- 결정: 재학습 없이 추론만 개선. 평가는 직선거리 기준, 평가셋 n20(200개)·n50(100개) seed 1234 고정.
- 결과: NN 대비 n20 +0.4% → +13.2%, n50 −3.3% → +14.3% (Run-01·02).
