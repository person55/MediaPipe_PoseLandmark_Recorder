# Claude Loop 진행 상황 (작업 재개용)

작성: 2026-07-19 (Claude Code 세션)
브랜치: `feat/claude-loop-pose-landmark-improvement`
노션 기록: "🔄 Claude Loop — Pose Landmark 개선" 페이지 (현황 리포트 + 구조 진단 섹션 기록 완료)

## 완료된 작업

### 1. 환경 구성 (Windows)
- Python 3.11.9 venv 생성, `pip install -e .` + pytest 설치 완료
- Blender 5.2 확인: `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` (PATH 없음 → `--blender-bin` 필요)
- Windows에서는 MediaPipe Python GPU delegate 미지원 → `--delegate cpu` 사용

### 2. 파이프라인 검증 (두 영상 모두 정상 완주)
- `session_cpu_006`: 3,324프레임, 검출률 96.4%, 전 레이어 정합, .blend 5.8MB
- `session_cpu_007`: 1,570프레임, 검출률 88.2%, 전 레이어 정합, .blend 2.5MB
- crop refine 수용 0건(006: 0/1,345, 007: 0/795), full-frame refine 수용 <1%

### 3. 구조 진단 (4관점 병렬 리뷰 완료, 노션에 상세 기록)
치명적 발견 4가지:
1. minimizer가 pose_world만 보정/break → export는 pose 원본을 읽어 **최종 출력에 미반영**
2. 재검출 스코어링 구조 결함 → 수용이 수학적으로 불가능 (delta 상한 ~0.04 < margin 0.06/0.08)
3. spike 판정: 6×median(MAD/하한 없음), root-relative 좌표, 스칼라 미분, 에코+3프레임 상한 → hip/ankle 오판 break
4. 종횡비 미적용(가로 24% 왜곡), 깊이 부호 역전 의심, exporter↔Blender 계약 파탄

## Loop 1: 배선 수정 + 종횡비 — 완료 (판정: 유지, 노션 [Loop 1] 기록 완료)

### 변경 파일
- `src/dance_pose_recorder/outlier_minimizer.py`: `sync_sources` 옵션(기본 true) 추가. pose_world 판정(보정/break)을 mirror 소스(pose)에 전파. `SOURCE_POSITION_FIELDS` 상수, `_interpolate_segment`에 fields 파라미터.
- `src/dance_pose_recorder/trajectory_exporter.py`: `apply_aspect_ratio` 옵션(기본 true). metadata width/height로 `screen_width_scale = height_scale × (W/H)` 파생. report에 aspect_ratio/requested 기록.
- `scripts/minimize_pose_outliers.py`: `--sync-sources/--no-sync-sources` 플래그
- `scripts/export_trajectory.py`: `--apply-aspect-ratio/--no-aspect-ratio` 플래그
- `scripts/open_blender_trajectory.py`: `--x-factor` 기본값 2.2 → 1.0 (aspect가 export에서 적용되므로)
- `tests/test_outlier_minimizer.py`: sync 3개 테스트 추가 (correct/break/disable)
- `tests/test_trajectory_exporter.py`: aspect 2개 테스트 추가

### 검증 결과 (113개 테스트 전체 통과 + 실데이터 재실행)
재실행 산출물: `examples/output/<sid>/outlier_minimized_loop1/`, `trajectory_export_loop1/`

| 지표 | 006 (old→new) | 007 (old→new) |
|---|---|---|
| pose 소스 corrected | 0 → 1,013 | 0 → 480 |
| pose 소스 break | 3,310 → 9,631 | 1,502 → 4,099 |
| export 내 corrected 점 | 0 → 1,013 | 0 → 480 |
| export segments (거짓 연결 제거) | 59,043 → 53,501 | 25,805 → 23,461 |
| X/Z 스팬 비율 | 2.307 → 4.102 | 1.974 → 3.509 |

- 비율 증가 = 정확히 1.778 (16:9 aspect) ✓
- pose_world 판정/좌표는 old와 byte-identical (회귀 없음) ✓
- 비보정 pose 행 좌표 무변경 ✓
- 참고: 007에서 pose hidden_unreliable 5,355→5,351 (4행이 missing_long_gap 겹침 spike로 break 전환 — world 쪽과 동일 semantics로 정렬된 것, 회귀 아님)

### Loop 1 마무리 확인 (2026-07-19)
1. `blender_session_cpu_006_trajectory_loop1.blend` 생성 확인 ✓
2. 판정: **유지(keep)** — 근거 위 표
3. 노션 [Loop 1] 기록 완료 ✓
4. 커밋 완료 ✓

## Loop 2: spike 임계값 재설계 — 완료 (판정: 유지, 노션 [Loop 2] 기록 완료)

### 변경 내용
- `temporal_features.py`: 가속/저크를 속력 스칼라 차분 → 벡터 2차/3차 차분 norm으로 (방향 전환 감지)
- `outlier_minimizer.py`:
  - `_feature_medians` → `_feature_scales`: scale = max(median + 1.4826×MAD, 절대 하한). 하한 옵션 `velocity_floor=0.02`, `acceleration_floor=0.02`, `jerk_floor=0.03` (pose_world m/frame 단위)
  - 베이스라인에서 `interpolated_short_gap` 행 제외 (`BASELINE_EXCLUDED_FLAGS`)
  - 에코 트리밍 `trim_feature_echo`(기본 true): velocity 스파이크가 있는 run에서 velocity 정상인 accel/jerk-only 프레임을 세그먼트에서 제외. 안정 이웃/보간 앵커 판정을 ratio 기반 → 최종 스파이크 집합 비포함 기준으로 변경(에코 프레임은 위치가 정상이므로 앵커 가능)
- `trajectory_policy.py`: hip 보정 허용 (`CORRECTABLE_TORSO_LANDMARKS = {left_hip, right_hip}`)
- CLI: `--velocity-floor/--acceleration-floor/--jerk-floor`, `--trim-feature-echo/--no-trim-feature-echo`
- 테스트 6개 추가(에코 트리밍 2, 노이즈 하한, hip 보정, 보간 베이스라인 제외, 벡터 가속), 다수-스파이크 합성 테스트 2개는 스파이크가 희소하도록 데이터 현실화. 총 119개 통과.

### 검증 결과 (재실행: `outlier_minimized_loop2/`, `trajectory_export_loop2/`)
| 지표 | 006 (loop1→loop2) | 007 (loop1→loop2) |
|---|---|---|
| spike 세그먼트 | 3,361 → 664 | 1,563 → 310 |
| world spike-break 행 | 5,527 → 521 | 2,170 → 276 |
| hip/ankle spike-break 행 | 1,274 → 37 (−97%) | 488 → 47 (−90%) |
| export 세그먼트 | 53,501 → 58,400 | 23,461 → 25,420 |

- 드롭된 스파이크 절대속도 중앙값 1.7~1.8 m/s(정상 동작 수준), 유지된 스파이크 4.7~7.0 m/s(글리치 수준) — 분리 명확
- old 상위 10개 velocity-ratio 세그먼트 전부 loop2에서도 break/corrected로 처리됨 (실제 스파이크 유지)
- 후속: preview 영상과의 시각 대조는 수동 확인 항목으로 남김

## Loop 3 (pending): 재검출 스코어링 공정화
- 계획: cleaned의 공짜 crop_valid=1.0 제거, temporal 점수 gap 정규화(앵커 거리로 나눔), crop z 복원 버그 수정(`crop_refine_pose.py:466-479`, z에 `bbox.w/frame_width` 스케일), margin 재캘리브레이션
- 검증: 저장된 `crop_refine_candidate_scores.csv`로 오프라인 재시뮬레이션 후 crop 스테이지 실재실행

## 재개 방법
1. 이 문서와 노션 페이지 확인
2. Loop 1 마무리(위 4단계) → Loop 2 → Loop 3
3. 각 루프: 가설 1개, 최소 변경, 기존 세션으로 전후 비교, 유지/보류/되돌림 판정, 노션 [Loop N] 기록, 커밋
