# ASCR AI Collaboration Log

This file is the shared handoff notebook between the local Codex session and
the server-side assistant/session. Update it whenever one side changes files,
runs validation, submits Slurm jobs, or discovers a blocker.

Do not write secrets, real API key values, private passwords, or full credential
paths into this file. Use environment variable names and redacted paths where
needed.

## How To Use This Log

Each handoff should append a new dated entry with:

1. actor: local Codex, server AI, or human;
2. Git branch and commit hash before work;
3. Git branch and commit hash after work, if changed;
4. files changed and why;
5. commands run, with concise results;
6. server environment details relevant to reproducibility;
7. Slurm job ids and log file paths;
8. generated output folders and important result files;
9. exact errors or warnings that need the other side to inspect;
10. next requested action for the other side.

If output is long, summarize the important lines and point to the log path. Do
not paste huge logs or generated data into this file.

## Entry Template

```text
## YYYY-MM-DD HH:MM TZ - actor

Context:
- Machine:
- Branch before:
- Commit before:
- Branch after:
- Commit after:

Files changed:
- path: reason

Commands run:
- command
  Result: passed/failed/skipped
  Notes:

Environment:
- python:
- torch:
- cuda:
- gpu summary:
- active env:
- important env vars set/unset, without values:

Server jobs:
- job id:
- mode:
- command:
- output dir:
- stdout log:
- stderr log:
- status:

Results:
- summary:
- files to inspect:

Problems / blockers:
- item:

Next action requested:
- item:
```

## Log Entries

No server run has been recorded yet.
