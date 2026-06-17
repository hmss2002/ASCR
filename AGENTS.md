# AGENTS.md

## Default workflow for this repository

This is my ASCR research codebase. The standard workflow is:

1. Modify the local repository.
2. Run reasonable lightweight validation.
3. Commit the completed changes.
4. Push the completed changes to GitHub automatically.
5. Tell me exactly how to continue on the university server.

## Local Python environment

Prefer a project-local virtual environment over installing dependencies into
Anaconda `base` or the system Python. On Windows, create it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/bootstrap_local.ps1
```

Activate it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/activate_local.ps1
```

When practical, run local validation through `.venv` or the currently active
project environment. Heavy CUDA/model dependencies should usually stay on the
remote Linux GPU server rather than this Windows laptop.

## GitHub auto-sync policy

At the end of every substantial task, Codex must automatically sync the result to GitHub.

Required steps:

1. Run:
   - git status --short --branch
   - git diff --stat

2. Check that no secrets, API keys, tokens, model weights, checkpoints, large outputs, local caches, or private data are staged.

3. Run a lightweight secret/path scan where practical, for example searching for:
   - sk-
   - api_key
   - token
   - secret
   - password
   - OPENAI_API_KEY
   - HF_TOKEN
   - HUGGINGFACE_TOKEN
   - GOOGLE_API_KEY
   - ANTHROPIC_API_KEY

4. Add appropriate files:
   - git add <changed project files>
   - do not blindly add ignored outputs, checkpoints, cache folders, datasets, or large binary files.

5. Commit with a clear message.

6. Push to GitHub automatically.

Default push rule:
- Push to the current branch if it already tracks a remote branch.
- If the current branch has no upstream, push with:
  git push -u origin HEAD
- Never force-push.
- Never rewrite history.
- If push fails because authentication, network, or branch permissions are unavailable, stop and print the exact manual command I should run.

## Server handoff policy

After pushing, Codex must provide a final "Server next steps" section with exact commands.

The final report must include:

1. Git branch and commit hash that was pushed.
2. Remote GitHub URL.
3. Commands to run on the university server:
   - clone if the repo does not exist;
   - pull if the repo already exists;
   - checkout the correct branch;
   - create or activate the environment;
   - install dependencies;
   - set required environment variables;
   - run a smoke test;
   - run the main inference/evaluation/training command;
   - run multi-GPU command if supported;
   - submit Slurm job if applicable.

4. If multi-node or 8-GPU execution is not fully implemented, clearly say what is supported now and what remains to be implemented.

## Safety constraints

- Do not modify files outside this repository.
- Do not commit secrets.
- Do not commit checkpoints, model weights, large outputs, datasets, cache folders, or local environment folders.
- Do not force-push.
- Do not delete important source files unless explicitly requested.
- Prefer simple, readable, debuggable changes.
- Preserve the intended research logic unless the existing code is clearly broken.
