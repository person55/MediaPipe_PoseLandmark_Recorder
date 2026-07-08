# Documentation Policy

## Goal

Keep documentation useful for humans while reducing Codex CLI context usage.

The project may have large planning documents, experiment logs, and research notes, but Codex should not read them by default.

## Documentation Layers

### 1. Always-Read Layer

These files must stay short:

```text
AGENTS.md
CURRENT_STATE.md
docs/codex_context_min.md
```

Recommended length:

```text
AGENTS.md: under 100 lines
CURRENT_STATE.md: under 100 lines
codex_context_min.md: under 150 lines
```

### 2. Task-Specific Layer

Read only when relevant:

```text
docs/quality_flags.md
docs/segment_refinement.md
docs/skeleton_optimization.md
docs/next_development_plan.md
docs/sample_reports/*.md
```

### 3. Archive Layer

Do not read by default:

```text
docs/archive/**
```

Archive candidates:

- long Notion exports
- old prompts
- research notes
- session logs
- obsolete implementation plans
- full experiment reports

## Codex Prompt Rule

Every Codex task should include:

```text
Read only:
- AGENTS.md
- CURRENT_STATE.md
- docs/codex_context_min.md
- task-specific files

Do not read:
- examples/output/**
- examples/input/**
- .venv/**
- docs/archive/**
- upstream MediaPipe source files
```

## Output Data Rule

Generated data should remain outside Git.

Never commit:

```text
examples/input/**
examples/output/**
models/*.task
*.mp4
*.mov
*.jsonl generated from sessions
raw/cleaned/refined/optimized CSV generated from sessions
```

## DB-Style Documentation

Use index-like documents instead of one large document.

Recommended structure:

```text
docs/
  README.md
  codex_context_min.md
  documentation_policy.md
  current_pipeline.md
  quality_flags.md
  segment_refinement.md
  skeleton_optimization.md
  next_development_plan.md
  sample_reports/
  task_prompts/
  archive/
```

## Rule of Thumb

If a document is useful for historical context but not needed for the next task, move it to `docs/archive/` or summarize it into `CURRENT_STATE.md`.
