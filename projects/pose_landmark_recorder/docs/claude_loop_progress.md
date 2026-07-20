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

## Loop 3: 재검출 스코어링 공정화 — 완료 (판정: 유지, 노션 [Loop 3] 기록 완료)

### 변경 내용
- `pose_candidate_scorer.py`: temporal 점수 gap 정규화 — 앵커까지 거리를 프레임 gap으로 나눈 per-frame rate로 판정. `_median_motion`도 per-frame rate 기준으로 통일. (보간값이 기하학적으로 항상 이기던 구조 해소)
- `crop_refine_pose.py`: `_fast_temporal_score`/`_temporal_references` 동일 정규화, `_restore_pose_landmarks`에 z 스케일 복원(`z × bbox.w/frame_width`) 추가
- `refine_pose_segments.py`: `_median_motion_for_series` gap 정규화
- margin 재캘리브레이션: 오프라인 재시뮬레이션 기준 노이즈 delta p99 ≈ 0.03 → crop 0.06→**0.04**, full-frame 0.08→**0.05** (options/argparse/pipeline_runner 3곳 모두)
- 테스트 2개 추가(gap 불변성, per-frame rate 판정). 총 121개 통과.

### 검증 결과 (crop: `crop_refine_loop3/`, full-frame: `refined_loop3/` 실재실행)
| 지표 | 006 (old→new) | 007 (old→new) |
|---|---|---|
| 달성 가능 delta 상한(오프라인) | 0.038 → 0.084 | 0.031 → 0.038 |
| crop 수용 | 0/1,345 → **11**/1,345 | 0/789 → 0/789 |
| full-frame 수용 | 56 → 76 | 57 → 93 |
| 후보 z 절대값 p90 | 0.47(crop 스케일) → 0.16 | 0.46 → 0.16 |

- z 수정 후 후보 z 분포가 cleaned z 분포(p90 0.16/0.18)와 정확히 일치
- crop 수용 11건 전부 목표 카테고리(estimated_occluded_arm 7, unreliable 4)
- 007 crop 수용 0건은 후보가 실제로 더 낫지 않은 정직한 결과 (수용이 "가능"해졌고 강제되지 않음)
- 가드 기각(review_only 6,615/5,726, missing candidate 3,834/2,851)은 old와 완전 동일 — 회귀 없음

## 후속 작업 — 완료 (2026-07-19)

### 1. preview 시각 대조 ✓
kept spike 구간(006 카트휠 f951-958·바닥 구르기 얼굴 f1339-1345, 007 바닥 런지 f789-798)은 전부 실제 글리치 유발 상황(신체 반전·좌우 혼동·얼굴 이탈), dropped 구간(wrist f1670, foot f360 등)은 실제 빠른 동작으로 확인. 판정 분류 타당.

### 3. 구조 정리 ✓ (커밋 ab581990f)
- `quality_flags.py`: stable/reliable/protected 플래그 집합 단일화 (5곳 → 1곳)
- `stage_schema.py`: crop/refine/outlier 스테이지 컬럼 계약 + COORD_FIELDS 단일화
- `crop_apply.py`/`refine_apply.py`: 병합·수용 로직을 스크립트에서 src로 이동
- `test_stage_contracts.py` 계약 테스트 추가, 총 123개 통과. 저장된 loop3 후보 재실행으로 동작 불변 확인(11/0 수용 재현)

### 2. 전체 파이프라인 재실행 ✓ (`session_cpu_006_v2`, `session_cpu_007_v2`)
record→cleaned→crop→refined→outlier→export→Blender 전 스테이지 정상 완주. 루프 검증 수치 재현:
- crop 수용 11/0 (margin 0.04), full-frame 수용 76/93 (margin 0.05)
- spike 세그먼트 664/308, floors (0.02, 0.02, 0.03) + 에코 트리밍 적용 확인
- export aspect_ratio 1.778, width_scale 6.0→10.67 파생 확인
- .blend 생성 (6.1MB / 2.6MB), 전 레이어 행 수 정합 (219,384 / 103,620)

## 검증 신뢰성 평가 (2026-07-19, Claude Code 작성)

### 입증된 것
"세션 006/007에서 파이프라인이 설계 의도대로 작동하게 됐다"는 주장은 반증 가능한 근거로 입증됨:
- Loop 1: X/Z 스팬 비율 증가가 정확히 1.778(16:9), pose_world byte-identical
- Loop 2: 드롭/유지 스파이크의 절대속도 분리(1.8 vs 5~7 m/s) + 시각 대조에서 양쪽 분류 모두 실상황과 일치
- Loop 3: 수정 후 후보 z 분포가 cleaned 분포와 정확히 일치, 가드 기각 수치 old와 완전 동일
- 전체 파이프라인 재실행에서 루프 검증 수치 재현

### 유보 사항 (아직 입증 안 된 것) — 전부 우리 쪽 방법론/설계 한계
1. **홀드아웃 부재**: margin(0.04/0.05)·floor(0.02/0.02/0.03)를 검증에 쓴 같은 두 세션에서 도출. 같은 무대·솔로 2명 = 사실상 표본 2개. 제3의 영상 검증 필요.
2. **floor의 fps 종속성**: m/frame 단위 절대값이라 60fps에서는 실질 2배 엄격해짐. m/s 정의 + fps 환산으로 바꿔야 함.
3. **수용 후보의 위치 품질 미검증**: 점수·가드 통과는 확인했지만 수용 좌표가 실제로 더 정확한지 영상 대조는 안 함 (full-frame 수용 증가 +20/+36 포함).
4. **시각 대조 표본 5건**: 드롭된 행의 속도 꼬리(12~15 m/s)에 거짓 음성 가능성 전수 확인 안 됨.
5. **구조 진단 발견 4의 잔여**: 깊이 부호 역전 의심 미조사, Blender 임포터가 exporter의 alpha/width 페이드 정책을 무시하고 재유도하는 문제 미해결 — 불확실 구간이 확실 구간과 동일하게 렌더됨.

### MediaPipe 자체의 한계 (파이프라인 철학이 대응하는 대상, 우리가 못 고치는 것)
- 반전·교차 자세에서의 좌우 혼동(학습 분포 밖), pseudo-depth의 단안 노이즈, landmark별 보정된 위치 오차 추정치 부재(visibility/presence는 대리 지표), crop 재검출의 낮은 기대 수익(전체 프레임 실패 원인이 크롭에도 대부분 유지)
- 원본 보존·break·비생성 철학은 이 한계에 대한 올바른 대응. 한계 자체를 밀어내려면 temporal 모델/다른 추정기가 필요하나 경량 단일 도구 노선을 의도적으로 유지 중.

### 다음 루프 우선순위 (검증 약점을 메우는 순서)
1. 제3의 영상(다른 환경/fps)으로 floor·margin 홀드아웃 검증
2. floor의 fps 정규화 (m/s 단위화)
3. Blender 임포터의 페이드 정책 계약 복원
- 그 외 후보: 대상 전환 징후 진단 정보(유일한 미구현 항목), Motion Profile Builder, AGENTS.md 현행화

## Loop 4: spike floor의 fps 정규화 — 구성됨 (2026-07-20, 구현 전)

### 관찰한 증상
- Loop 2에서 도입한 spike floor(velocity 0.02 / accel 0.02 / jerk 0.03)가 **m/frame 절대값**. 같은 실제 동작이라도 프레임당 이동량은 fps에 반비례하므로, 60fps 입력에서는 velocity 기준이 실질 2배(0.6→1.2 m/s), accel은 fps², jerk는 fps³으로 왜곡됨.
- 결과적으로 고fps 영상에서는 글리치 미검출(거짓 음성), 저fps 영상에서는 정상 동작 오판이 경고 없이 발생할 수 있음.
- 이 왜곡은 다음 우선순위인 홀드아웃 검증(제3의 영상, 다른 fps)과 교락되므로 홀드아웃보다 먼저 해소해야 함.

### 가설
floor를 물리 단위(m/s, m/s², m/s³)로 정의하고 로드 시 metadata fps로 프레임 단위 환산하면, 기존 세션(≈30fps)의 판정은 사실상 불변으로 유지되면서 어떤 fps 입력에서도 동일한 물리 기준이 적용된다.

### 변경 범위 (최소)
- `outlier_minimizer.py`: floor 옵션을 초 단위로 정의하고 내부에서 fps로 환산 (velocity: /fps, accel: /fps², jerk: /fps³). fps는 이미 `outlier_minimizer.py:105`에서 metadata로부터 로드됨 — `max_correction_gap_sec × fps`와 동일 패턴.
- 기본값은 30fps 등가로 설정: velocity 0.6 m/s, accel 18 m/s², jerk 810 m/s³ (= 기존 0.02/0.02/0.03 × 30/30²/30³).
- CLI 플래그를 초 단위 명칭으로 전환(기존 m/frame 플래그는 deprecated 또는 제거 — 구현 시 결정).
- 테스트: ① fps=30에서 기존 floor와 수치 동등, ② 동일 물리 궤적을 30/60fps로 샘플링한 합성 데이터에서 spike 판정 집합 동일, ③ fps 결측 시 기본 30 fallback.

### 기준선과 비교 방법
- `session_cpu_006_v2` / `session_cpu_007_v2` 산출물을 기준선으로 outlier 스테이지 오프라인 재실행.
- 세션 실제 fps가 정확히 30이면 byte-identical 기대, 아니면(예: 29.97) 환산 차이로 인한 판정 변화가 0 또는 무시 가능한 수준인지 spike 세그먼트/break 행 수로 확인.
- 60fps 검증은 합성/리샘플 데이터로 수행 (실제 60fps 영상은 홀드아웃 루프에서).

### 판정 기준
- 유지: 30fps 기준선 판정 불변(또는 환산 오차 수준) + 합성 60fps에서 fps 불변성 입증 + 테스트 통과.
- 원칙 정합: 임계값 정의 변경일 뿐 모션 생성/원본 덮어쓰기 없음 → 노션 3장 원칙과 무충돌.

### 후속 루프 후보 (Loop 4 이후)
- Loop 5: 제3의 영상 홀드아웃 검증 — **사용자 제공 영상 필요** (다른 환경/fps 권장, 60fps면 이상적)
- Loop 6: Blender 임포터 페이드 정책 계약 복원 + 깊이 부호 검증

## 재개 방법
1. 이 문서와 노션 페이지 확인
2. Loop 1~3 + 후속 작업 + 문서 현행화(AGENTS/CURRENT_STATE/next_development_plan, 2026-07-20) 완료. Loop 4(위 계획)부터 진행
3. 각 루프: 가설 1개, 최소 변경, 기존 세션으로 전후 비교, 유지/보류/되돌림 판정, 노션 [Loop N] 기록, 커밋
