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

## Loop 4: spike floor의 fps 정규화 — 완료 (2026-07-20, 판정: 유지)

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
- 유지: 기준선 판정 불변(또는 환산 오차 수준) + 합성 60fps에서 fps 불변성 입증 + 테스트 통과.
- 원칙 정합: 임계값 정의 변경일 뿐 모션 생성/원본 덮어쓰기 없음 → 노션 3장 원칙과 무충돌.

### 구현 결과 (2026-07-20)
- **계획 수정 1건**: 기준 세션의 실제 fps는 30이 아니라 **23.976**. floor의 검증된 물리값은 0.02 m/frame × 23.976 = 0.48 m/s이므로 30fps 등가(0.6 m/s)가 아닌 23.976fps 등가를 기본값 앵커로 채택.
- `OutlierMinimizerOptions`: `velocity_floor_m_per_s=0.48`, `acceleration_floor_m_per_s2=11.5`, `jerk_floor_m_per_s3=414.0` (반올림으로 인한 프레임 단위 오차 +0.03~0.13%). 신규 `_frame_floors()`가 fps로 환산(/fps, /fps², /fps³)해 `_feature_scales`에 전달.
- CLI: `--velocity-floor-m-per-s` / `--acceleration-floor-m-per-s2` / `--jerk-floor-m-per-s3` (기존 m/frame 플래그 제거). 리포트 settings에 초 단위 값 + 파생 프레임 단위 값 모두 기록.
- 테스트 3종 추가 (총 126개 통과): ① 동일 물리 동작(4 m/s 버스트, floor 지배 베이스라인)을 30/60fps로 샘플링 시 spike 판정 동일 + ratio 오차 <5% — 구 코드라면 60fps에서 ratio가 절반이 되어 미검출, ② 리포트 환산값 정확성, ③ fps 결측 시 30 fallback.

### 검증 결과 (outlier_minimized_loop4/ 오프라인 재실행, 기존 outlier_minimized/와 전수 비교)
| 지표 | 006_v2 | 007_v2 |
|---|---|---|
| spike 세그먼트 | 664 → 663 | 308 → 308 (동일) |
| outlier_status 변경 행 | 4 / 219,384 (0.002%) | 0 / 103,620 |
| 좌표 변경 행 | 1행(×2소스, 보정 전환분) | 0 (byte-identical) |

- 007_v2: 산출물 완전 동일 (status·좌표 byte-identical).
- 006_v2 변화 4행은 전부 임계 경계 케이스로 확인: right_ankle f2262 accel ratio 6.0012→5.9996(경계 6.0을 0.02% 차이로 탈락), 이로 인해 f2260이 안정 이웃을 얻어 break→corrected로 개선. right_heel f2788은 jerk ratio 8.0076→7.9975로 spike_type만 mixed→accel 변경(status 불변). 5~7 m/s급 실제 글리치에는 영향 없음.
- 판정: **유지** — 기준선 보존(환산 오차 수준) + fps 불변성 입증 + 원본 보존·비생성 원칙 무관.

### 후속 루프 후보 (Loop 4 이후)
- Loop 5: 제3의 영상 홀드아웃 검증 — **사용자 제공 영상 필요** (다른 환경/fps 권장, 60fps면 이상적. fps 정규화 완료로 이제 fps 교락 없이 검증 가능)
- Loop 6: Blender 임포터 페이드 정책 계약 복원 + 깊이 부호 검증

## Loop 6: Blender 페이드 정책 계약 복원 + 깊이 부호 교정 — 완료 (2026-07-20, 판정: 유지)

Loop 5(홀드아웃)는 제3의 영상 미확보로 보류하고 Loop 6을 선행.

### 관찰한 증상 (구조 진단 발견 4의 잔여)
1. 임포터가 exporter의 `trajectory_alpha`/`trajectory_width`를 완전히 무시 — 모든 트레일이 고정 alpha(overview 0.30, progress 0.95)·고정 굵기로 렌더되어 불확실 구간이 확실 구간과 동일하게 보임. 실데이터에서 페이드 세그먼트 1,121건(006_v2) / 708건(007_v2)이 전부 뭉개짐.
2. 깊이 부호 역전 실증: MediaPipe pose z는 "작을수록 카메라에 가까움"(docs/solutions/pose.md:220 + **영상 대조로 확인**: 006 f1711 z=-0.33 → 손목이 가슴 앞 카메라 쪽, f1309 z=+0.27 → 손목이 몸 뒤). exporter가 `blender_y = -z`로 반전하고 Blender 카메라는 -Y에서 +Y를 보므로, 카메라에 가까운 부위가 Blender에서 더 멀리 렌더됨.

### 가설
exporter의 깊이 부호를 z 그대로 유지하고(`blender_y = z × depth_scale`), 임포터가 alpha/width를 재유도 없이 소비하면(0.1 단위 버킷), 로컬 깊이 방향이 물리적으로 올바르게 되고 불확실 구간이 시각적으로 구분된다.

### 변경 범위
- `blender_coordinate.py`: `blender_y = z × depth_scale` (부호 교정). 임포터의 로컬 깊이는 `(raw_y − root)` 상대값이라 코드 무변경으로 방향이 함께 교정됨.
- `open_blender_trajectory.py`: 세그먼트의 alpha/width를 0.1 단위로 버킷팅해 티어별 커브 객체 분리. overview alpha = 0.30×α, progress emission strength·alpha ×α, bevel 굵기 ×width. 티어 경계에서 draw path 분할. JSON 요약에 `fade_policy_tiers` 통계 추가. alpha=1.0 티어는 기존과 동일 머티리얼 재사용(솔리드 구간 시각 회귀 없음).
- `docs/trajectory_export.md` 좌표식 갱신, `test_blender_coordinate.py` 부호 테스트 갱신.

### 기준선과 비교 방법 / 결과
- 재-export(`trajectory_export_loop6/`) 전수 비교: **blender_y(y1/y2)만 정확히 부호 반전, 나머지 전 컬럼 byte-identical** (006_v2 points 86,885 / segments 58,408; 007_v2 38,003 / 25,434).
- Blender 5.2 헤드리스 임포트: 006_v2 티어 6종 소비(a0.5~a1.0, 합계 58,408 = 세그먼트 수 일치), overview 커브 21→94 객체(티어 분리), .blend 6.3MB 생성. 007_v2 동일 검증(티어 6종, .blend 2.8MB).
- 테스트 126개 전체 통과.

### 판정: 유지
- 부호 교정은 문서 규약 + 영상 실증 양쪽으로 근거 확보. 페이드 소비는 exporter 계약의 순수 시각 반영(모션 데이터 무변경, 원본 보존·비생성 원칙 무관).

### 남은 한계 (기록)
- 애니메이션 마커/halo는 여전히 프레임별 alpha를 반영하지 않음 (트레일만 페이드 소비). 필요 시 후속 루프.
- 프레임 단위 전신 깊이는 여전히 임포터가 신체 크기 휴리스틱으로 유도 — 이는 계약 위반이 아니라 설계(pose z는 hip 상대 로컬 깊이뿐이므로 전역 깊이는 exporter에 없음).

## Loop 7: One-Euro 시각화 스무딩 레이어 — 완료 (2026-07-20, 판정: 유지)

### 배경
Loop 1~6은 정확성 수리였고 소진폭 잔떨림(특히 pseudo-depth z 노이즈)은 원칙상 어느 스테이지도 건드리지 않아 Blender 궤적에 그대로 유입됨. 사용자 시각 확인에서 이 문제가 지적됨.

### 가설
export 단계에 confidence-인지형 One-Euro 필터(저속 강한 스무딩·고속 추종)를 **별도 컬럼 레이어**로 추가하고 break/gap에서 필터를 리셋하면, 원본 보존·비생성 원칙을 지키면서 시각 떨림이 제거된다.

### 변경 범위
- `trajectory_smoothing.py` 신설: One-Euro 필터 (min_cutoff/beta/d_cutoff), 체인 단위 적용.
- `trajectory_exporter.py`: points에 `blender_{x,y,z}_smooth`, segments에 `{x,y,z}{1,2}_smooth` 컬럼 추가. 체인 판정(연속 프레임 + connect)이 끊기면 필터 리셋 — 스무딩이 break를 이어붙이지 않음. 원본 blender 좌표는 무변경 유지. 기본 파라미터: x/z min_cutoff 1.2Hz·beta 1.5, 깊이(y) 0.4Hz·beta 0.4 (깊이가 가장 노이즈가 크므로 강한 스무딩).
- `export_trajectory.py` CLI 플래그 6종, `open_blender_trajectory.py`: `*_smooth` 컬럼 존재 시 우선 사용(`--no-use-smoothed-trajectory`로 해제), 트레일·마커·포인트클라우드 모두 적용.
- 테스트 7종 추가(필터 4: 상수 통과/노이즈 감소/고속 지연 한계/리셋, exporter 3: 지터 감소·비활성화·break 리셋), 총 133개 통과.

### 검증 결과 (trajectory_export_loop7/ 재-export, loop6 대비)
| 지표 | 006_v2 | 007_v2 |
|---|---|---|
| 깊이(blender_y) 지터(2차 차분) | **−88%** | **−87%** |
| x/z 지터 (전체) | −14~16% | −13% |
| x/z 노이즈 (저속 프레임 한정) | −24% | — |
| 고속 프레임 지연 (p95) | 0.12~0.19 유닛 (≈1~2프레임 모션) | 동일 수준 |
| 원본 blender 좌표 | loop6과 byte-identical | 〃 |

- 해석: 시각 떨림의 지배 요인이던 깊이 노이즈가 ~88% 제거. x/z의 잔여 2차 차분은 대부분 실제 저속 모션(보존 대상)임을 저속 한정 측정으로 확인. 고속 동작 지연은 1~2프레임 수준으로 One-Euro의 적응 특성이 작동.
- Blender 5.2 헤드리스 임포트로 smoothed 소비 확인, `.blend` 생성(loop7, 6.3MB).

### 판정: 유지
원본 좌표 무변경(byte-identical) + 스무딩은 분리 컬럼 + break 리셋으로 거짓 연결 없음 → 원칙 무충돌. 파라미터는 두 세션 기준 초기값이며 홀드아웃 검증 대상에 포함.

## 연구 스파이크 결론 및 파기 (2026-07-20)

사전학습 VideoPose3D(키포인트 lifting) 스파이크 결과: 표면 지표(지터 1/5, 글리치 구간 연속성)는 압도적이었으나 **관측 충실도 검사 탈락** — 프레임별 완전 affine 적합에서도 출력이 관측 2D와 신체 크기의 50% 어긋남(우리 pose_world 5.9%), 안정 구간 포함 전 구간. 즉 매끄럽지만 지어낸 모션. 사용자 결정: **정식 도입 기각, 스파이크 파기** (브랜치 `research/motion-prior-spike` 로컬·원격 삭제, 작업물 제거). 상세 기록은 노션 "연구 스파이크" 섹션에 보존. 이후 개선은 아래 Loop 8~10 — 전부 "실제 픽셀에 대한 실제 검출" 후보 소스 추가로, 생성 없이 관측 자체를 개선하는 경로.

## 정비 패키지: 코덱스 v2 리포트 P0 권고 이행 (2026-07-20, 루프 외)

코덱스 "v2 데이터 비교·개선 리포트"의 권고 중 제3 영상 홀드아웃(영상 미확보)을 제외한 P0 3건을 이행:

### 1. CSV 무결성 회귀 테스트 (`test_trajectory_export_integrity.py`, 4종)
코덱스 수동 검사를 고정 계약으로 자동화 — point 복합 키 유일성/null 없음, segment endpoint↔point 정합(raw+smooth), 스무딩의 raw 컬럼 불변, 모든 chain start의 smooth=raw. 이후 루프가 export 계약을 건드릴 때의 안전망.

### 2. 재현성 manifest (`session_manifest.py` + `write_session_manifest.py`)
세션별 `manifest.json`: 입력 영상 sha256, Git commit/branch/dirty, 환경(model/delegate/fps/해상도), 스테이지별 리포트 설정·수치 요약, 핵심 CSV sha256+행 수. 파이프라인 러너 마지막 스테이지로 통합. 006_v2/007_v2에 생성 완료. 홀드아웃 검증의 절차 요건(사전 기록) 충족 기반.

### 3. 267개 좌표 변경 + 662개 제거 segment 영상 오버레이 검토 (코덱스 필수 caveat 해소)
- **267행 좌표 변경**: 변위 중앙값 4.8px(보수적 미세 보정), p90 24px, 최대 183px. 62개 클러스터, 팔다리 말단 집중. 최대 변위 프레임 시각 확인 — f1347(바닥 눕기): 기존 발/힐 마커가 몸 밖 허공 → 새 좌표는 신체 위. f2162: elbow가 머리 위치 오검출 → 몸통 쪽으로 교정. **타당한 보정으로 판정.**
- **662개 제거 segment**: elbow/wrist 집중(팔 = 기지의 불안정 그룹), 제거분 길이 중앙값 0.066으로 유지분(0.024)의 ~2.8배 — 큰 점프를 잇던 연결. 66%(436)는 명시적 trajectory_break 정책의 결과. 최장 제거 4건 시각 확인 — 전부 몸 밖 글리치 점들을 잇던 연결(예: f2572 right_wrist, 양 끝점이 신체 밖 허공). **올바른 제거로 판정.**

나머지 권고 배치: raw/smooth 전환은 임포터 플래그로 기구현(Blender 내 토글은 시각화 마무리 루프로), marker/halo alpha는 시각화 마무리 루프로, 대상 전환 진단은 Loop 10에서 해소 예정, 홀드아웃은 영상 확보 시 최우선.

## Loop 8: 회전 증강 재검출 — 완료 (2026-07-20, 판정: 유지)

### 구현 (계획 대비 변경 2건 포함)
- `crop_rotation.py` 신설: hip→shoulder 신체 축 각도 → 60° 이상 기울면 90° 단위 스냅(90=머리 오른쪽→CCW, 180, 270=머리 왼쪽→CW), cv2 회전, 좌표·월드 벡터 역회전(실제 cv2 회전과의 라운드트립 테스트로 수학 검증), z 스케일은 검출기가 본 이미지 폭 기준으로 보정.
- 반전 세그먼트 타겟팅: 기존 세그먼트 미커버 반전 프레임 run → `inverted_pose_segment` (문제 플래그에 measured 포함 — 반전 혼동은 확신에 찬 오검출이므로).
- **중간 발견 1 (계획 수정)**: 회전 검출로 기존 검출을 "대체"하면 기존 수용 5건이 소실됨 → 정립+회전 **이중 후보 경쟁**으로 전환 (`crop_apply` 다중 후보 지원, 최고 after_score 선택).
- **중간 발견 2**: 단일 video-mode 추적기에 신규/회전 프레임을 섞으면 이후 프레임의 추적 상태가 오염되어 회전 비대상 프레임(f1673 등)의 후보까지 변함 → **추적기 3개 분리**: A=기존 세그먼트 프레임만(기준선과 byte-동일 스트림), B=회전 검출, C=신규 반전 세그먼트 정립 검출.

### 검증 결과 (crop_refine_loop8/, 기존 crop_refine/ 대비)
| 지표 | 006_v2 | 007_v2 |
|---|---|---|
| 기존 수용 보존 | **11/11** | 0/0 (해당 없음) |
| 수용 행 | 11 → **189** | 0 → **135** (최초 수용) |
| 수용 중 회전 후보 | 116 (90°:86, 270°:29, 180°:1) | 59 (90°:35, 180°:22, 270°:2) |
| 회전 검출 프레임 | 292 | 204 |
| 반전 세그먼트 수용률 | 171/17,463 (~1%, 강제 아님) | 122/11,786 |
| 바닥 구르기(1330-1360) 수용 | 34행 | — |

- 시각 확인: f1347에서 cleaned가 몸 밖 허공에 두던 발/힐 좌표(정비 3에서 확인된 실패 지점)가 270° 회전 후보로 다리 위에 안착. 테스트 145개 전체 통과.
- 판정: **유지** — 목표 구간 수용 발생 + 회전 후보가 실수용의 주력 + 기준선 완전 보존 + 수용은 여전히 스코어러·가드 통과분만.

### 남은 한계 (기록)
- 카트휠 f951-958의 measured 행은 기존 mixed 세그먼트 소속이라 수용 대상 밖(세그먼트 문제 플래그에 measured 없음) — 그 구간은 여전히 Loop 2 break가 담당. 기존 세그먼트의 수용 대상 확대는 별도 루프 후보.
- 다운스트림(refine→outlier→export) 통합 재실행은 Loop 9~10 완료 후 일괄(v3) 예정.

## Loop 8 계획: 회전 증강 재검출 — 반전 자세 좌우 혼동 공략 (원계획)

### 관찰한 증상
- 카트휠(006 f951-958)·바닥 구르기(006 f1339-1345)·바닥 런지(007 f789-798)에서 반전/수평 자세의 좌우 혼동 → spike break. MediaPipe가 정립 자세 분포로 학습되어 반전 입력이 분포 밖인 것이 원인(스파이크 시각 대조로 확인된 실패 목록).

### 가설
문제 세그먼트의 크롭을 신체 축(hip 중점→shoulder 중점 벡터) 기준으로 정립화 회전해 재검출하고 랜드마크를 역회전 복원하면, 모델 입력이 분포 안으로 들어와 더 나은 후보가 생기고, 기존 스코어러(Loop 3 공정화)와 가드 하에서 수용된다.

### 변경 범위 (최소)
- crop refine에 회전 후보 소스 추가: ① cleaned hip/shoulder로 프레임별 신체 축 각도 산정(90° 단위 스냅부터 시작 — 보수적), ② 크롭 회전(cv2.rotate) 후 검출, ③ 랜드마크 역회전 복원(x/y 회전, z는 crop 스케일 복원 로직 유지), ④ 후보 출처 컬럼 `candidate_source=rotated_crop` 추가. 스코어링·가드·margin은 무변경.
- 신체 축이 정립(±45° 이내)인 프레임은 회전 후보 생성 생략(비용 절약 + 회귀 위험 차단).

### 기준선과 비교 방법
- 006_v2/007_v2 crop refine 스테이지 재실행(`crop_refine_loop8/`). 목표 세그먼트에서의 수용 수, 수용 후보의 영상 프레임 대조(기존 후속작업 방식), 정상 구간 회귀(기존 수용 11/0 및 가드 기각 수치 유지).

### 판정 기준
- 유지: 목표 세그먼트 수용 발생 + 영상 대조로 좌표 타당 + 정상 구간 무회귀. 수용 0이어도 가드가 막은 정직한 결과면 기록 후 보류 판정 가능(후보 품질 분석 첨부).

## Loop 9 계획: 크롭 전처리 개선 — 저조도 대비 향상

### 관찰한 증상
- 무대 저조도(스파이크 프레임 추출로 확인). 어두운 구간에서 visibility/presence 하락과 검출 실패.

### 가설
재검출 크롭에 CLAHE(+필요시 감마) 대비 향상을 적용한 후보(`enhanced_crop`)를 추가하면 저조도 구간 검출 품질이 올라 수용이 늘어난다. 전처리는 검출기 입력에만 적용 — 원본 영상·좌표 무변경이므로 정책 무풍.

### 변경 범위 / 비교
- crop refine 후보 소스 1종 추가(Loop 8과 같은 틀). 저 visibility 구간 한정 수용·스코어 분포 비교. Loop 8 결과와 독립 검증(각각 별도 재실행).

## Loop 9: 크롭 CLAHE 저조도 rescue — 완료 (2026-07-20, 판정: 유지)

### 구현
- `crop_enhancement.py`: LAB L-채널 CLAHE(clip 2.0, tile 8) — 검출기 입력에만 적용, 원본 영상·좌표 무변경.
- rescue 트리거: 해당 프레임의 모든 검출(정립+회전)이 약할 때만(전신 평균 visibility < 0.5 또는 무검출) CLAHE 크롭 재검출을 추가 후보로 투입. 전용 추적기(스트림 격리 원칙 유지). `crop_enhanced` 컬럼으로 출처 기록.

### 검증 결과 (crop_refine_loop9/, loop8 대비)
| 지표 | 006_v2 | 007_v2 |
|---|---|---|
| loop8 수용 보존 | **189/189** | **135/135** |
| rescue 트리거 프레임 | 2 | 3 |
| enhanced 후보 수용 | 0 | **+4** (f614 elbow, f1146 손끝) |

- 시각 확인: f1146(몸 접은 자세) 손끝 좌표 개선 확인. 테스트 148개 통과(enhancement 3종 추가).
- 판정: **유지** — 회귀 0 + 비용 근사 0(프레임 2~3개 추가 검출). 단 **효과는 제한적**임을 정직 기록: 전신 평균 visibility가 0.5 미만으로 떨어지는 프레임 자체가 이 세션들에선 드묾. 가치는 향후 더 어두운 입력(홀드아웃 후보 포함)에 대한 보험 성격.

## Loop 10: 역방향/미러 패스 + 혼동 진단 — 완료 (2026-07-20, 판정: 유지)

### 구현
- `crop_mirror.py`: 미러 검출(크롭 좌우 반전 → 검출 → x 역반전 + 좌우 landmark 정체성 스왑, 월드 x 부호 반전) — 스왑 인볼루션·라운드트립 테스트로 검증. 프레임별 정/미러·정/회전 불일치도(`pass_disagreement`)와 `possible_confusion` 플래그.
- 역방향 패스: 세그먼트 단위 크롭 버퍼 → 세그먼트 경계에서 역순 재검출(전용 추적기, 합성 단조 타임스탬프) — 추적기가 반대 시간 방향에서 진입.
- 패스별 전용 추적기(총 6개: baseline/inverted-upright/rotated/enhanced/mirror/reverse)로 스트림 격리 유지. `crop_pass` 출처 컬럼. `crop_confusion_diagnostics.csv` + 리포트 요약(진단은 메타데이터 전용, 자동 보정 없음).

### 검증 결과 (crop_refine_loop10/, loop9 대비)
| 지표 | 006_v2 | 007_v2 |
|---|---|---|
| loop9 수용 보존 | **189/189** (소실 0) | **139/139** (소실 0) |
| 수용 행 | 189 → **1,284** | 139 → **789** |
| 수용 출처 | mirror 895 / reverse 320 / forward 69 | reverse 373 / mirror 306 / forward 110 |
| 혼동 플래그 | 89프레임/21런 — **카트휠 f952-968 정확히 포함** | 51프레임/13런 — 런지 인접 f767-784 포함 |

- 시각 확인: f2279(깊은 런지)에서 가슴에 뭉쳐 있던 손 랜드마크가 미러 후보로 실제 바닥 짚은 손 위치로, 역방향 후보가 다리 위로 정착.
- **대상 전환 징후 진단(노션 5장의 유일한 미구현 후보) 해소**: 정/미러 불일치도가 기지의 좌우 혼동 구간(카트휠)을 threshold 0.03에서 정확히 플래그. 순수 관측 메타데이터로, 표시만 하고 복원하지 않음.
- 테스트 153개 전체 통과. 판정: **유지**.

### 남은 한계 / 후속
- 신규 수용 1,095/650행은 스코어·가드 통과 기준이며 위치 정확도는 표본 시각 확인 수준 — 코덱스 P1(층화 표본 검증)이 이 수용분에도 적용됨.
- 검출 비용 ~2.5× (문제 세그먼트 한정이므로 절대량은 수 분 수준).
- 다음: v3 통합 재실행(crop_loop10 산출물로 refine→outlier→export→Blender 전체) + 층화 정확도 검증 + 홀드아웃(영상 확보 시).

## Loop 10 계획: 역방향/미러 패스 — 추적 관성 해소 + 혼동 진단 (원계획)

### 관찰한 증상
- video 모드 추적기가 오류를 시간 순방향으로 끌고 감(글리치가 세그먼트화되는 원인 일부). 좌우 혼동에는 모델 비대칭 요인도 존재.

### 가설
(a) 문제 세그먼트를 역순 프레임으로 재검출하면 추적기가 반대 방향에서 진입해 다른 잠금 후보를 얻는다. (b) 좌우 미러 입력 검출(좌표·좌우 라벨 재반전) 후보가 혼동 구간에서 유효하다. (c) 정/역/미러 검출 간 불일치도가 "대상 전환·혼동 징후" 진단 지표가 된다 — 노션 5장의 유일한 미구현 후보를 순수 관측 메타데이터로 해소.

### 변경 범위 / 비교
- 세그먼트 재검출에 pass 방향/미러 옵션 + 프레임별 불일치도 컬럼과 리포트. 후보 수용은 기존 가드 하에서만, 진단 지표는 수용과 무관하게 기록.

### 공통 규율
- 루프당 가설 1개씩 순차 진행(8 → 9 → 10), 각 루프 완료 시 전후 비교·판정·노션 [Loop N] 기록·커밋.
- 홀드아웃 검증(구 Loop 5)은 제3의 영상 확보 시 최우선으로 삽입.

## v3 통합 재실행 + 층화 정확도 검증 — 완료 (2026-07-20)

### v3 통합 재실행 (Loop 8~10 crop 산출물 → refine→outlier→export→Blender)
산출물: `refined_v3/`, `outlier_minimized_v3/`, `trajectory_export_v3/`, `blender_*_v3.blend` (양 세션).

| 지표 | 006_v2 (v2→v3) | 007_v2 (v2→v3) |
|---|---|---|
| full-frame refine 수용 | 76 → 28 | 93 → 32 |
| spike 세그먼트 | 664 → 676 | 308 → 324 |
| outlier corrected 행 | 500 → 518 | 116 → 152 |
| 최종 export 연결 세그먼트 | 58,408 → **58,657** (+249) | 25,434 → **25,591** (+157) |
| export 포인트/hidden | 불변 | 불변 |

- 해석: crop 단계가 좋아지자 full-frame refine의 수용이 자연 감소(이길 여지 축소). spike/break 소폭 증가는 "매끄럽게 틀렸던" 값이 정확한 값으로 바뀌며 이웃 행과의 경계 전이가 생긴 것으로, 대부분 corrected로 처리됨. 최종 연결 세그먼트 순증 = 궤적 연결성 개선.
- 배선 검증: 수용 crop pose 행 중 export에 노출되는 412행의 **93%(382)가 좌표 그대로 최종 도달**, 30행은 다운스트림 재경쟁으로 정당 대체. (미노출분은 Blender 프리셋 제외 랜드마크.)

### 층화 정확도 검증 (코덱스 P1)
- **객관 지표(전수, 교차 패스 합의)**: 수용 좌표와 독립 패스 검출의 최근접 거리 — 007 중앙값 0.0086(≈16px), 77%가 0.02 이내 합의(≥2개 독립 합의 135행). 006 중앙값 0.0233, 45% 합의(혼동 심한 반전 구간 특성). 양 세션 모두 수용 좌표가 대체된 cleaned 값보다 독립 검출들에 더 가까움(방향성 지지).
- **층화 시각 표본(세션×패스 12장, 댄서 확대)**: 전 표본 온바디·타당. 대변위 사례는 전부 "가슴 높이에 떠 있던 손 랜드마크 → 실제 바닥 짚은 손" 교정. 반대편 오지정·몸 밖 수용 0건.
- 잔여 유보: 006의 교차 패스 합의 45%는 "독립 검출들이 서로 흩어지는 어려운 구간에서의 수용"이 존재함을 의미 — 홀드아웃 검증과 함께 재점검 대상.

## 홀드아웃 검증 1차 (구 Loop 5): session_cpu_008 — 완료 (2026-07-21, 판정: 통과)

### 입력
`dance_take_008.mp4` — 1920×1080, 23.976fps, 3,254프레임(~2분 16초). 다른 무대 촬영분. **주의: fps가 기존 세션과 동일(23.976)이라 fps 일반화는 이번에도 미검증** — 환경/동작 홀드아웃으로만 유효. 영상 중간에 카메라 무빙 컷 편집으로 인한 인물 위치 점프 1회 존재(사용자 고지, 사전 등록된 검증 포인트).

### 전체 파이프라인 (`session_cpu_008`, 설정 무변경 기본값)
- 7개 스테이지 전부 정상 완주, `.blend` 5.6MB, manifest 기록(commit ef69bd80b, dirty=false). 검출률 91.3%(2,970/3,254 — 006 96.4%와 007 88.2% 사이). 전 레이어 행 수 정합(214,764).
- spike floor fps 환산 정확(0.0200/frame), aspect 1.778 적용, One-Euro 스무딩 컬럼 생성. 사용자 Blender 시각 확인: 직관적으로 잘 구현됨.

### 카메라 컷 점프 처리 — 원칙대로 동작 (핵심 검증 포인트)
- 컷 위치 식별: f2096→f2139, 43프레임 검출 결측 후 hip 중심 변위 0.443(화면의 ~44%, 정상 동작 최대 점프 0.015의 ~30배).
- 결측 42프레임 전부 `missing_long_gap` → `hidden_unreliable`, **보간 0건** (15프레임 상한 초과분은 채우지 않는 정책 작동).
- **컷을 가로지르는 export 세그먼트 0건** — 점프를 잇는 거짓 궤적 없음. 기타 장기 결측 4구간(115/24/37/26프레임)도 전부 연결 0건.
- One-Euro 필터가 컷 직후 f2139에서 리셋(전 27포인트 smooth==raw) — 스무딩이 컷을 이어붙이지 않음.

### Loop 8~10 패스 일반화
- 회전 후보 232프레임, CLAHE rescue 8프레임(006/007의 2~3보다 많음 — 더 어두운 구간 존재), 미러/역방향 패스 각 ~875프레임, 혼동 진단 84프레임 플래그.
- crop 수용 1,560행(pose 소스 821행) — v3(1,284/789)과 같은 규모, 기각 11,095행 대비 강제 수용 양상 없음. full-frame refine 수용 15행(v3 경향과 일치: crop이 좋아지면 full-frame 여지 축소). spike 세그먼트 612, 보정 408행, break 968건.

### 교차 패스 합의 분석 (수용 pose 821행 전수, `crop_refine/crosspass_agreement.csv`)
- 최근접 독립 패스 거리 중앙값 **0.0111**, 72.1%가 0.02 이내 합의 — 007(0.0086/77%)과 006(0.0233/45%) 사이의 양호한 수준.
- 패스별: reverse 232행 합의 90.9%(변위 중앙값 0.005 — 가장 안전), forward 74행 83.8%, mirror 515행 61.9%(혼동 구간 특성).
- **해석 주의(정직 기록)**: "수용 좌표가 대체된 cleaned 값보다 독립 검출에 더 가까움" 비율은 36.5%로 v3의 방향성 지지와 다름. 원인은 mirror 수용이 다수인 구조 — 좌우 혼동 구간에서는 독립 패스들도 cleaned와 같은 혼동을 공유하므로 이 지표가 편향됨. 시각 표본에서 저합의 수용(f1409 d=0.077, f2488 d=0.131)이 모두 시각적 정답으로 확인되어, 이 지표는 혼동 구간에서 단독 판정 기준이 될 수 없음을 실증.

### 층화 시각 표본 (11장: mirror 5 / forward 3 / reverse 3, heel·foot 포함, `verification_samples/`)
- **11/11 온바디 또는 인접, 몸 밖·반대편 오지정 0건.**
- 대변위 사례 전부 명백한 올바른 교정: f484(허벅지에 있던 검지→머리 위 든 손, 독립 합의 0.000), f1914(골반에 떠 있던 손끝→바닥 짚은 손, 0.001), f1962(목 근처의 발끝→접힌 실제 발), f2488(턱 높이 검지→늘어뜨린 손).
- 경계 사례 1건: f1982 left_heel(reverse) — cleaned보다 발에 가까워졌으나 발끝에서 약간 어긋남. heel 그룹은 합의 지표도 최저(right_heel 43.8%) — 후속 루프 후보.

### 판정: **통과** — margin/floor/스무딩/신규 패스 전부 미지의 영상에서 설정 무변경으로 설계 의도대로 동작. 잔여: fps 일반화(60fps 영상 필요), heel/foot 수용 품질.

## Loop 12: 교차 패스 합의 리포트 표준화 — 완료 (2026-07-21, 판정: 유지)

Loop 11보다 선행 실행 (Loop 11 검증의 측정 도구가 되므로).

### 가설
홀드아웃 검증에서 수작업으로 수행한 교차 패스 합의 분석을 crop refine 스테이지의 표준 산출물로 내장하면, 수용 로직 무변경(순수 관측 메타데이터)으로 모든 세션에서 수용 품질을 자동 감시할 수 있다.

### 변경 범위
- `src/dance_pose_recorder/crosspass_agreement.py` 신설: `build_crosspass_agreement(refined, candidates, cleaned)` — 수용 pose 행별 최근접 독립 패스 거리, cleaned 대비 거리, 변위 + 요약(전체/패스별/랜드마크별). 모듈 docstring과 리포트 `note`에 해석 주의(혼동 구간 편향 — 단독 판정 기준 사용 금지)를 명문화.
- `crop_refine_pose.py`: 스테이지 산출물로 `crop_crosspass_agreement.csv` 기록 + 리포트 JSON에 `crosspass_agreement` 요약 삽입. `output_layout.py`에 `CROP_REFINE_CROSSPASS_CSV` 상수.
- `scripts/report_crosspass_agreement.py` 신설: 기존 세션 백필용(재검출 없이 저장된 candidates/pose/cleaned CSV에서 생성, `--crop-refine-dir`로 loop 변형 디렉토리 지정 가능, 리포트 JSON 갱신 포함).
- 테스트 4종 추가 (`test_crosspass_agreement.py`): 최근접 독립 거리·cleaned 비교·per-pass 요약, 독립 패스 부재 시 NaN(합의로 오인 금지), 수용 pose 행만 측정, 빈 입력 계약. 총 157개 통과.

### 검증 결과
- **session_cpu_008 백필 = 수작업 분석과 완전 일치**: 821행, 중앙값 0.0111, 합의 72.1%, closer-than-cleaned 36.5%, 패스별 forward 74/0.838·mirror 515/0.619·reverse 232/0.909 전부 동일.
- 006_v2/007_v2 (crop_refine_loop10) 백필: 0.0237/44.0%, 0.0099/74.6% — v3 검증 기록(0.0233/45%, 0.0086/77%)과 정합(소폭 차이는 당시 수작업 방법론 세부 차이 수준).
- 판정: **유지** — 관측 메타데이터 확장뿐, 수용·좌표 무변경. 회귀 0 (157 테스트).

## Loop 11: heel/foot 수용 가드 — 진단 후 보류 (2026-07-21, 판정: 보류)

### 관찰한 증상 (홀드아웃에서)
008에서 발 그룹 합의 최저(right_heel 43.8%, right_foot_index 25%, right_ankle 50%) + 경계 시각 표본 1건(f1982 left_heel). 가설: 발 전용 가드(margin 상향 등)로 경계 수용을 걸러낼 수 있다.

### 진단 결과 — 가설 반증 (Loop 12 산출물로 3세션 전수 분석)
1. **score delta는 저합의/고합의 발 수용을 구분하지 못함**: 발 저합의 delta 중앙값 vs 고합의 — 008: 0.061/0.054, 006_v2: 0.060/0.060, 007_v2: 0.055/0.050. margin 상향은 좋은 수용과 나쁜 수용을 무차별 제거.
2. **visibility/presence도 판별력 없음** (풀링 148행: 0.856 vs 0.838 / 0.999 vs 0.999). displacement는 분리되나(0.050 vs 0.020) 대변위가 곧 최고 가치 교정(f484·f2488 유형)이라 가드로 쓰면 역효과.
3. **발 약세는 체계적 현상이 아님**: 006_v2 발 합의 58% > 전체 43%, 007_v2 발 77% ≈ 전체 74%. 008만 발(65%) < 전체(73%)이고, 저합의 17행 중 13행이 f1941–1982 바닥 기어가기 구간에 집중 — 시각 표본 ⑨⑩에서 수용이 정답/개선으로 확인된 바로 그 구간. 낮은 합의는 독립 검출이 흩어지는 어려운 구간의 신호이지 잘못된 수용의 증거가 아님.

### 판정: **보류** — 근거 없는 가드 추가는 무차별 데이터 손실. 코드 무변경. Loop 12의 랜드마크별 합의 통계(`per_landmark_rows_and_agreement`)가 세션마다 자동 기록되므로, 이후 세션들에서 발 그룹 약세가 체계적으로 재현되면 그 데이터로 재개.

## 코덱스 교차 검증 반영 (2026-07-21, 노션 "v2 데이터 비교·개선 리포트" > Codex 교차 검증 섹션)

코덱스가 Loop 12 구현·3세션 백필을 독립 재현(행별 대조 완전 일치, 157 passed)하고 시각 표본 11장을 직접 재확인. 아래 교정·보강을 기록에 반영:

- **Loop 11 결론 표현 교정**: "가설 완전 반증"이 아니라 **"현재 대리 지표(score delta/visibility/presence/displacement)로는 가드를 지지하지 못함"**이 정확한 결론. 놓쳤던 분석 축: 발 저합의율의 패스별 차이 — forward 51.6%(16/31), mirror 42.6%(26/61), reverse 14.3%(8/56). 즉시 가드 규칙은 아니나 "판별력 없음"을 완결 결론으로 만들지 않음.
- **합의 지표 방법론 주의 보강**(모듈 note에 반영): forward/mirror/reverse는 동일 모델·동일 이미지 계열의 탐색 경로 차이로 통계적 독립 표본이 아니며, 다수 후보 중 최근접값 사용은 거리를 낙관적으로 작게 만듦. "cleaned보다 가깝다"도 정확도 증명이 아닌 방향성 보조지표. 지표는 **교차 패스 일치도(consistency)**로 읽을 것.
- **Loop 11 재개 프로토콜**(재개 시 대리 지표 중앙값 비교 대신): ① 블라인드 시각 라벨(정답/경계/오류)을 landmark·패스·난동작 구간별 층화 수집, ② 후보 특징(패스, rotation/enhancement, confusion flag, crop edge risk, 국소 속도·가속·jerk, ankle-heel-foot bone/접지 일관성) 기록, ③ frame/segment 상관을 고려한 구간 단위 신뢰구간·PR/AUC로 판별력 평가.
- **다음 우선순위 합의**: ① marker/halo 프레임별 fade(exporter alpha 계약 기존재, 관측·수용 정책 무변경의 저위험 시각화 완결), ② 60fps 홀드아웃(fps 일반화), ③ Motion Profile Builder는 **read-only 통계 리포트로 시작** — 현 검증 세션이 전부 23.976fps·소표본이라 acceptance threshold에 연결하면 과적합 위험; 가드 승격은 더 다양한 세션과 라벨 검증 이후.

## Loop 13: marker/halo 프레임별 페이드 — 완료 (2026-07-21, 판정: 유지)

코덱스 교차 검증에서 합의된 다음 우선순위 1번. 관측·수용 정책 무변경의 시각화 완결 작업.

### 관찰한 증상 (Loop 6의 남은 한계)
트레일은 exporter의 trajectory_alpha/width 페이드 계약을 소비하지만(Loop 6), 애니메이션 마커/halo는 고정 alpha로 렌더 — 불확실 프레임의 마커가 확실 프레임과 동일하게 보임.

### 가설
마커/halo 머티리얼에 Object Info(alpha) 노드를 추가하고 오브젝트 color alpha를 exporter의 trajectory_alpha로 키프레임하면, 공유 머티리얼을 유지한 채 마커가 프레임별로 페이드되고, alpha=1 프레임은 기존과 동일하게 렌더된다.

### 변경 범위
- `src/dance_pose_recorder/marker_fade.py` 신설: alpha 파싱(결측/NaN→1.0, [0,1] 클램프) + 변화 지점 압축(`build_marker_alpha_events`, CONSTANT 보간 전제) — bpy 없이 단위 테스트 가능.
- `open_blender_trajectory.py`: ① marker/halo 머티리얼(그룹·좌우 8종)에 `object_alpha` 노드 경로 추가 — Object Info alpha → Mix Shader(fac)로 Transparent와 혼합 + emission 강도에도 alpha 곱(포화로 인한 약한 페이드 비가독 보완). **alpha=1이면 두 경로 모두 원본 셰이더 통과(수학적 동일)**. ② 마커·halo 오브젝트에 color alpha 키프레임(변화 지점만, CONSTANT). ③ `--no-marker-frame-fade` 플래그(기본 on). ④ JSON 요약에 `marker_frame_fade` 통계. 트레일 머티리얼 무변경.
- 테스트 5종 추가 (`test_marker_fade.py`), 총 162개 통과.

### 검증 결과 (session_cpu_008, Blender 5.2 헤드리스)
- 임포트: 페이드 대상 8,419 point 행 → 3,788 alpha 키프레임(변화 지점 압축), min alpha 0.2, `blender_..._loop13.blend` 생성.
- fcurve 실측: f1985에서 left_knee color alpha **0.25**, 나머지 0.85 — points CSV 값과 정확 일치. `--no-marker-frame-fade` 빌드는 전부 1.0.
- 렌더 전후 비교(f1985, 27개 랜드마크 페이드): 페이드 마커의 뚜렷한 감광 확인(최대 픽셀 diff 85.3), 마커 배치 동일. emission 강도 스케일 추가 전 40.7 → 추가 후 85.3으로 가독성 2배. 증거: `verification_samples/loop13_f1985_marker_fade_{on,off}.png`.
- 참고: 2,982 프레임 전부에 페이드 랜드마크가 ≥1개 존재해 whole-frame 픽셀 동일성 검사는 불가 — alpha=1 경로의 무변경은 노드 수학(mix fac=1 통과, strength×1)으로 보장.

### 판정: **유지** — 페이드 계약의 마커 레벨 소비 완성(Loop 6 남은 한계 해소), 모션 데이터·수용 정책 무변경, 기본 상태 시각 회귀 없음.

### 남은 한계
- emission 포화 특성상 약한 페이드(0.85~0.9)는 여전히 은은함 — 강한 페이드(≤0.5)는 명확. 필요 시 후속에서 alpha의 비선형 매핑 검토 가능(계약 해석 변경이라 별도 판단).
- 오브젝트 color alpha는 렌더/머티리얼 프리뷰 모드에서 보임 — 솔리드 뷰포트 셰이딩은 반영 안 됨(뷰포트 한계, 기록).

## Loop 14: Motion Profile Builder (read-only) — 완료 (2026-07-21, 판정: 유지)

### 범위 (코덱스 합의 준수)
순수 관측 통계 리포트 — 어떤 파이프라인 스테이지도 이 프로파일을 임계값·수용 판정에 읽지 않음. 가드 승격은 세션 다양화 + 블라인드 라벨 검증 이후 별도 판단.

### 구현
- `src/dance_pose_recorder/motion_profile.py`: pose_world 안정 행(STABLE_MEASUREMENT_FLAGS)에서 연속 프레임 체인 한정 벡터 차분 → fps/fps²/fps³ 환산 물리 단위(m/s·m/s²·m/s³) 속도/가속/저크 + BONE_PAIRS 본 길이. 세션 풀링 + 세션별 요약.
- `scripts/build_motion_profile.py`: 세션별 레이어 경로 오버라이드(`dir::relpath`) 지원 → `configs/motion_profile_default.json` + `motion_profile_report.md`.
- 테스트 6종 (핵심: 동일 물리 동작의 30/60fps 샘플링 통계 동일성 = fps 불변성), 총 168개 통과.

### 관찰 (4세션: 006_v2 v3 / 007_v2 v3 / 008 / 009)
- **hip 속도 p95 = 0.44~0.52 m/s** — Loop 4에서 검증 도출한 velocity floor(0.48 m/s)와 독립 경로로 일치. floor 타당성의 통계적 뒷받침.
- 유지된 실제 글리치(5~7 m/s)는 wrist p99(5.2~5.3) 밖 영역 — spike 판정과 분포 정합.
- 교차 세션 p95 일관: wrist 3.34~3.81 m/s (60fps 포함).

## 홀드아웃 검증 2차 (60fps): session_cpu_009 — 완료 (2026-07-21, 판정: 통과)

### 입력
`dance_take_009.mov` — 1920×1080, **60.0fps**, 8,394프레임(~2분 20초). **다른 환경**(프로젝션 맵핑 무대, 어두운 배경). fps 일반화의 첫 실검증.

### 전체 파이프라인 (설정 무변경 기본값)
- 7 스테이지 완주, 전 레이어 정합(554,004행), .blend 13.2MB, manifest 클린(commit 3ea4f7e19). 검출률 81.0%(가장 어려운 영상). crop 수용 540, full-frame 38, aspect 1.778.
- **spike floor fps 환산 정확**: 0.48/60=0.008, 11.5/60²=0.0031944, 414/60³=0.00191667 — 리포트 실측 일치.
- Loop 12 합의 리포트 스테이지 자동 생성 확인: 278행, 중앙값 0.0092, **합의 78.8%(전 세션 최고)**, reverse 233/forward 9/mirror 36.

### fps 일반화 검증 (핵심)
1. **spike 판정의 물리 속도 분포가 24fps 세션과 동일**: 009 spike 인접 속도 중앙값 1.61 / p90 6.86 m/s vs 006_v2 1.59/7.40, 008 2.11/9.35. 정상 행도 동일(중앙값 0.51, p99 4.26 vs 0.49~0.69/3.93~4.30). 정규화가 깨졌다면 보였을 2.5배(=60/24) 왜곡 없음.
2. spike 세그먼트 수(3,716)는 24fps 세션 대비 프레임당 ~2.2배이나, **지속시간 중앙값 0.017초(=1프레임)** vs 008 0.083초 — 60fps가 24fps에서 뭉개지던 마이크로 글리치를 포착하는 샘플링 해상도 효과 + 검출률 81%의 영상 난이도. 시간 단위 보정 창(0.12s=7프레임)이 작동해 보정 2,370행(008 408행).
3. **Motion Profile 교차 검증**: 60fps 세션의 물리 단위 p95가 24fps 세션들과 같은 범위(wrist 3.81 vs 3.34~3.67, hip 0.436 vs 0.454~0.520) — fps 정규화(Loop 4)의 실데이터 최종 입증.

### 시각 표본 (3장, `verification_samples/`)
f3615 좌우 혼동 교정 명확(왼손 검지·소지가 반대편 손에 있던 것을 mirror 후보가 들어올린 실제 왼손으로, 변위 0.198), f5698 미세 조정 타당. 몸 밖·반대편 오지정 0건.

### 판정: **통과** — fps 일반화 입증(물리 단위 판정 불변). 잔여 유보: 009 수용 540행의 전수 위치 검증은 표본 수준(코덱스 P1 프로토콜 대상), 검출률 81% 영상의 spike 다발은 어려운 영상 특성으로 기록.

## 홀드아웃 검증 3차 (60fps 확인 런): session_cpu_010 — 완료 (2026-07-21, 판정: 통과)

### 입력
`dance_take_010.mov` — 1920×1080, **59.969fps**(비정수 fps), 585프레임(~9.8초). 사용자 고지: 쉽고 짧은 영상.

### 결과 (설정 무변경 기본값)
- 7 스테이지 완주, 전 레이어 정합(38,610행), .blend 1.2MB, manifest 클린(commit 3c4bc9821).
- **검출률 100%** (585/585), 숨김 행 0 — "쉬운 영상" 가설 그대로.
- **비정수 fps 환산 정확**: velocity floor 0.48/59.96925 = 0.0080041 — 리포트 실측 일치.
- crop 재검출 0건 시도(후보 세그먼트 3개 전부 선택 기준 미달), full-frame 수용 0 — 재검출이 불필요한 영상에서 아무것도 강제하지 않는 정직한 무위. Loop 12 합의 리포트의 빈 입력 계약이 실전 검증됨(0행 CSV + 요약 정상 기록).
- spike: 262 세그먼트(1,000프레임당 448 ≈ 009의 443), 지속시간 중앙값 0.017초 — **"60fps가 1프레임 마이크로 글리치를 해상한다"는 009 해석이 두 번째 60fps 영상에서 재현**. spike 인접 물리 속도 2.97/p90 10.08 m/s vs 정상 0.41/5.29 — 분리 명확.
- Motion Profile 5세션 갱신: hip p95 0.41~0.52 대역 유지(floor 0.48 지지 지속), wrist p95 3.34~3.81 일관. right_ankle p95 5.43은 짧은 클립의 강한 킥 특성(안무 편차)으로 기록.

### 판정: **통과** — 검출 품질 상한(100%) 영상에서도 파이프라인이 과잉 개입 없이 동작. 60fps 계열 spike 특성 재현으로 009의 해석 확정.

## 홀드아웃 검증 4차 (카메라 무빙): session_cpu_011 — 완료 (2026-07-21, 판정: 통과)

### 입력
`dance_take_011.mov` — 1920×1080, 60.0fps, 376프레임(~6.3초). 사용자 고지: **카메라 무빙 있음**, 신체 가림 최소. 발레(객석 뷰), 또 다른 환경. 카메라 모션이라는 새 스트레스 축의 첫 검증.

### 결과 (설정 무변경 기본값)
- 7 스테이지 완주, 레이어 정합(24,816행), .blend 0.7MB, manifest 클린. 검출률 **100%**, 숨김 0.
- crop 수용 123행(2세그먼트), **교차 패스 합의 95.1% — 전 세션 최고**. full-frame 2행.
- spike 168 세그먼트 = 447/1,000프레임 — 60fps 계열(009 443, 010 448) 세 번째 재현.

### 카메라 모션 강건성 분석 (핵심)
- 카메라 팬 실증: f237→238 배경(장비열·조명) 이동 확인. 화면 hip 이동 중앙값 0.0018/frame, 고이동 프레임 79/376(21%).
- spike 물리 분리 유지: spike 인접 2.89/p90 9.23 m/s vs 정상 0.55/p99 6.12 — 기존 세션들과 동일 프로파일. **hip 상대 pose_world 판정이 카메라 팬에 설계대로 강건.**
- spike 행이 고이동 프레임에 6배 농축(7.7% vs 1.3%) — 시각 판별 결과 **정당한 판정**: spike 밀집 f207은 점프-턴 중 등을 보인 순간으로, spike 랜드마크가 정확히 비가시 부위(등 돌린 얼굴, 몸 앞에 감춘 손)에 한정되고 가시 부위(다리·힙)는 정상 판정. "카메라가 따라가는 격동 순간 = 관측이 어려운 순간"의 상관이지 좌표 오염이 아님. 증거: `verification_samples/`.
- 설계 특성 기록: `screen_bottom_origin` 궤적(pose 소스)에는 카메라 팬이 인물 이동으로 혼입됨 — 결함이 아닌 현 좌표계의 알려진 특성. 카메라 모션 보상(배경 기준 안정화)은 별도 기능 후보.

### Motion Profile 6세션 갱신
hip p95 대역 유지, wrist 일관. 프로파일이 6세션(24fps×3, 60fps×3, 환경 5종)으로 확장.

### 판정: **통과** — 카메라 무빙 하에서도 판정 체계 강건, 합의 95.1% 최고 기록.

## session_cpu_009 수용분 전수 검증 (P1 프로토콜) — 완료 (2026-07-21, 판정: 가드 변경 불필요)

코덱스 권고 블라인드 층화 라벨 방식으로 009 수용 pose 행 **278개 전수** 시각 검증. 상세 문서: `docs/session_009_exhaustive_verification.md` (코덱스 교차 검증용).

- 결과: **정답 261 (93.9%) / 경계 17 (6.1%) / 오류 0**. 경계는 전부 랜드마크 크기 이내 오프셋(주먹 가장자리 등), 오지정·몸 밖·반대편 0건.
- 사례 연구 f8379: 몽타주 해상도의 유일한 오류 후보였으나 고배율 줌에서 실제 주먹 인접 확인 — cleaned(엉덩이 오지정) 대비 명확한 개선.
- 판별력: 경계 행의 독립 패스 거리 중앙값 0.0021 < 정답 0.0097 — **합의 지표가 정답성 판별자가 아니라는 결론의 전수 데이터 실증**. 오류 0건으로 PR/AUC는 양성 클래스 부재로 종결.
- 최적화 판단: **가드·margin 변경 불필요** — 걸러낼 오류가 없고 경계는 어떤 지표로도 구분 불가. 근거 없는 조정 안 함(Loop 11 보류와 동일 원칙). 코덱스 P1 caveat은 009에 대해 전수 수준으로 해소(006_v2/007_v2/008은 표본 수준 유지).
- 증거: `session_cpu_009/verification_samples/` (라벨 CSV, 몽타주 11장, 스크립트, f8379 줌).

## 전 세션 전수 검증 + 정비 패키지 + 머지 준비 — 완료 (2026-07-21)

코덱스 2차 리포트(009 전수 검증 재현·수용, 단 완전 독립 블라인드 아님 → κ 프로토콜 권고) 반영 후 실행.

### 1. 3세션 전수 시각 검증 (006_v2/007_v2/008, 총 1,938행 / 559프레임)
- 방법: 009와 동일 블라인드 몽타주(37장) + **기계 사전 스크리닝 신설**(수용 좌표가 cleaned 스켈레톤에서 0.06+ 이탈한 9행 자동 플래그 → 전건 줌 재판독).
- 결과: **정답 1,935 / 경계 3 / 오류 0**. 4세션 합산 2,216행 — 정답 99.1%, 경계 0.9%, **오류 0**. 플래그 9행은 8 정답 + 1 경계(전부 물구나무 발목·카트휠 손 등 최대 가치 교정). 합의 최저(44%)였던 006_v2도 오류 0 — "저합의 = 혼동 구간의 정직한 수용" 최종 확인.
- 판정: 가드·margin 변경 불필요 재확정. P1 caveat 전 세션 해소. 상세: `docs/acceptance_exhaustive_verification.md` (코덱스 2차용, κ 프로토콜 명시).
- 오독 교정 사례 기록: 008 f1750을 몽타주에서 힐로 오독 → 줌에서 핸드스탠드의 바닥 짚은 right_index 정답 확인 (몽타주 해상도 한계와 줌 재판독 절차의 필요성 재확인).

### 2. 정비 패키지
- **PyInstaller 빌드 재생성**: 기존 dist는 Loop 1 이전 코드 — 전체 루프 개선 반영해 재빌드, `--help` 스모크 통과.
- **v3 레이아웃 승격**: 006_v2/007_v2의 `*_v3`/`crop_refine_loop10` 산출물을 표준 디렉토리로 승격(기존 v2 산출물은 `*_preloop8/`로 보존), manifest 재생성.
- **AGENTS.md 현행화**: 파이프라인 목록(manifest 스테이지·진단 스크립트 포함), 브랜치 문구 갱신.

## 재개 방법 (2026-07-21 갱신)
1. 이 문서와 노션 페이지 확인
2. 완료: Loop 1~4, 6~10, 12~14 + 연구 스파이크 기각·파기 + 코덱스 P0 정비 + v3 통합 재실행 + 층화 정확도 검증(P1) + 홀드아웃 1차(008)·**2차(009, 60fps — fps 일반화 입증)** + 코덱스 교차 검증 반영. Loop 11은 보류(재개 시 블라인드 라벨 프로토콜). 다음 후보: 임포터 영속화(Blender add-on/TouchDesigner), 009 수용분 전수 검증(P1 프로토콜), Motion Profile 가드 승격 검토(세션 추가 축적 후)
3. 각 루프: 가설 1개, 최소 변경, 기존 세션으로 전후 비교, 유지/보류/되돌림 판정, 노션 [Loop N] 기록, 커밋
