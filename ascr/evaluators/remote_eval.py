"""File-IPC evaluator: delegates semantic evaluation to a separate process.

The MMaDA-8B generator (transformers 4.46) and the Qwen3.5-9B evaluator
(transformers 5.2.dev fork) cannot be loaded in the same Python process — MMaDA's
remote modeling code is incompatible with the newer transformers. To still run the
*original* coarse ASCR pipeline (MMaDA generator + Qwen-9B selector), the evaluator
lives in a separate resident process (``scripts/qwen_eval_server.py``) and this thin
adapter forwards each ``evaluate`` call to it over a shared IPC directory.

Protocol (atomic writes via tmp + rename):
  ``IPC/server_ready``       — written by the server once the 9B is loaded.
  ``IPC/requests/r{seq}.json`` — request payload written by this adapter.
  ``IPC/responses/r{seq}.json``— SemanticEvaluation.to_dict() written by the server.
  ``IPC/server_stop``        — sentinel asking the server to exit.
"""

import json
import os
import time
from pathlib import Path

from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.evaluators.base import SemanticEvaluator


def _evaluation_from_dict(payload, grid_size):
    if not isinstance(payload, dict):
        return SemanticEvaluation.abstain("Remote evaluator returned a non-object response")
    if payload.get("should_abstain"):
        return SemanticEvaluation.abstain(
            str(payload.get("parser_error") or payload.get("summary") or "remote abstain"),
            raw=payload.get("raw"),
        )
    regions = []
    for region in payload.get("regions", []) or []:
        cells = []
        for cell in region.get("cells", []) or []:
            try:
                cells.append(GridCell.from_any(cell, grid_size))
            except Exception:
                continue
        if cells:
            regions.append(RegionSelection(
                cells=cells,
                reason=str(region.get("reason", "")),
                confidence=float(region.get("confidence", 1.0)),
                error_type=str(region.get("error_type", "semantic")),
                action=str(region.get("action", "reopen")),
            ))
    has_error = bool(payload.get("has_error", False))
    if has_error and not regions:
        return SemanticEvaluation.abstain(
            str(payload.get("summary") or "remote reported error without regions"),
            raw=payload.get("raw"),
        )
    return SemanticEvaluation(
        has_error,
        summary=str(payload.get("summary", "")),
        regions=regions,
        correction_instruction=str(payload.get("correction_instruction", "")),
        raw=payload.get("raw"),
    )


class RemoteFileEvaluator(SemanticEvaluator):
    def __init__(self, ipc_dir, grid_size=4, request_timeout=900.0, ready_timeout=1800.0, poll_interval=0.5):
        self.ipc_dir = Path(ipc_dir)
        self.grid_size = int(grid_size)
        self.request_timeout = float(request_timeout)
        self.ready_timeout = float(ready_timeout)
        self.poll_interval = float(poll_interval)
        self.requests_dir = self.ipc_dir / "requests"
        self.responses_dir = self.ipc_dir / "responses"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def wait_for_server(self):
        ready = self.ipc_dir / "server_ready"
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if ready.exists():
                return True
            time.sleep(self.poll_interval)
        return False

    def stop_server(self):
        try:
            (self.ipc_dir / "server_stop").write_text("stop")
        except Exception:
            pass

    def _write_atomic(self, path, text):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        self._seq += 1
        name = f"r{self._seq:06d}.json"
        request = {
            "seq": self._seq,
            "original_prompt": original_prompt,
            "grid_image_path": str(Path(grid_image_path).resolve()),
            "iteration": iteration,
            "current_prompt": current_prompt,
            "grid_size": self.grid_size,
        }
        self._write_atomic(self.requests_dir / name, json.dumps(request))
        resp_path = self.responses_dir / name
        deadline = time.time() + self.request_timeout
        while time.time() < deadline:
            if resp_path.exists():
                try:
                    payload = json.loads(resp_path.read_text())
                except Exception:
                    time.sleep(self.poll_interval)
                    continue
                return _evaluation_from_dict(payload, self.grid_size)
            time.sleep(self.poll_interval)
        return SemanticEvaluation.abstain(f"Remote Qwen evaluation timed out after {self.request_timeout}s")
