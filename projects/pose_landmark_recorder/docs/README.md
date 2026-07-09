# Documentation Index

This folder is organized as a lightweight documentation database.

Codex should usually read only:

```text
../AGENTS.md
../CURRENT_STATE.md
codex_context_min.md
```

Task-specific documents should be read only when needed.

## Core Context

| Document | Purpose | Read frequency |
|---|---|---|
| [`../AGENTS.md`](../AGENTS.md) | Agent rules, safety, reading policy | Always |
| [`../CURRENT_STATE.md`](../CURRENT_STATE.md) | Current project status | Always |
| [`codex_context_min.md`](codex_context_min.md) | Minimal project context for Codex | Always |
| [`documentation_policy.md`](documentation_policy.md) | Documentation structure and archive policy | Occasionally |

## Pipeline Docs

| Document | Purpose |
|---|---|
| [`quality_flags.md`](quality_flags.md) | Meaning of quality flags and display policy |
| [`segment_refinement.md`](segment_refinement.md) | Segment re-detection workflow |
| [`skeleton_optimization.md`](skeleton_optimization.md) | Conservative skeleton diagnostics |
| [`next_development_plan.md`](next_development_plan.md) | Next planned modules and priorities |

Skeleton optimization is not the default final visualization stage.
For visualization-first work, prefer the refined/outlier-minimized path and use skeleton optimization as diagnostic context.

## Recommended Reading by Task

| Task | Read |
|---|---|
| Basic extraction | `codex_context_min.md`, main README |
| Cleaning and interpolation | `quality_flags.md`, `current_pipeline.md` if available |
| Segment re-detection | `segment_refinement.md` |
| Visualization-oriented cleanup | `next_development_plan.md`, future `outlier_minimization.md` |
| Skeleton diagnostics | `skeleton_optimization.md` |
| Blender import | future `blender_importer.md` |

## Sample Reports

| Document | Purpose |
|---|---|
| [`sample_reports/session_003_refine_summary.md`](sample_reports/session_003_refine_summary.md) | Small summary of session_gpu_003 refine results |

## Archive

Long notes, old prompts, research dumps, and Notion exports should go under:

```text
docs/archive/
```

Codex should not read `docs/archive/**` unless explicitly instructed.

## Task Prompts

Reusable task prompts should go under:

```text
docs/task_prompts/
```

Each task prompt should specify:

- files to read
- files not to read
- task scope
- test command
- commit scope
