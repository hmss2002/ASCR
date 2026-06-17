from ascr.core.artifacts import RunArtifacts, current_git_commit, runtime_manifest
from ascr.grids.overlay import create_token_grid_overlay
from ascr.revision.prompt_composer import compose_correction_prompt
from ascr.traces.schema import make_trace_record
from ascr.traces.writer import TraceWriter


class DirectTokenReopenLoop:
    """Stage 1 loop variant that reopens discrete image tokens directly.

    This mirrors :class:`ascr.core.loop.ASCRLoop` but renders a token-resolution
    reference overlay (``create_token_grid_overlay``) instead of the 4x4 coarse
    overlay, so the evaluator selects which of the ``token_grid_size`` x
    ``token_grid_size`` discrete image tokens are wrong and those exact tokens
    are reopened (no coarse downsample, dilation, or upsampling). It is a
    separate module so the original :class:`ASCRLoop` stays unchanged.
    """

    def __init__(self, generator, evaluator, selector, config, label_step=4):
        self.generator = generator
        self.evaluator = evaluator
        self.selector = selector
        self.config = config
        self.label_step = int(label_step)

    def run(self, prompt, project_root=".", initial_state=None):
        artifacts = RunArtifacts.create(self.config.output_dir, self.config.run_name)
        artifacts.write_json("config_snapshot.json", {
            "run_name": self.config.run_name,
            "stage1_variant": "direct_token",
            "max_iterations": self.config.max_iterations,
            "image_size": self.config.image_size,
            "coarse_grid_size": self.config.coarse_grid_size,
            "token_grid_size": self.config.token_grid_size,
            "label_step": self.label_step,
            "git_commit": current_git_commit(project_root),
            "started_from_initial_state": initial_state is not None,
        })
        artifacts.write_json("runtime_manifest.json", runtime_manifest(project_root))
        trace_writer = TraceWriter(artifacts.root / "trace.jsonl")
        state = initial_state if initial_state is not None else self.generator.initialize(prompt, artifacts)
        records = []
        evaluator_calls = 0
        stop_reason = "max_iterations"
        current_prompt = prompt
        final_decoded_image = None
        final_grid_image = None
        initial_decoded_image = None
        initial_grid_image = None
        raw_final_decoded_image = None
        raw_final_grid_image = None
        for iteration in range(self.config.max_iterations):
            iteration_dir = artifacts.iteration_dir(iteration)
            decoded_path = iteration_dir / "decoded.ppm"
            state = self.generator.decode(state, decoded_path)
            final_decoded_image = str(decoded_path)
            raw_final_decoded_image = str(decoded_path)
            grid_path = iteration_dir / "grid.ppm"
            create_token_grid_overlay(decoded_path, grid_path, image_size=self.config.image_size, token_grid_size=self.config.token_grid_size, label_step=self.label_step)
            final_grid_image = str(grid_path)
            raw_final_grid_image = str(grid_path)
            if iteration == 0:
                initial_decoded_image = str(decoded_path)
                initial_grid_image = str(grid_path)
            evaluation = self.evaluator.evaluate(prompt, str(grid_path), iteration, current_prompt=current_prompt)
            evaluator_calls += 1
            mask = self.selector.select(evaluation)
            evaluation_path = artifacts.write_json(f"iterations/{iteration:03d}/evaluation.json", evaluation.to_dict())
            mask_path = artifacts.write_json(f"iterations/{iteration:03d}/reopen_mask.json", mask.to_dict())
            artifact_paths = {
                "decoded_image": str(decoded_path),
                "grid_image": str(grid_path),
                "evaluation": str(evaluation_path),
                "reopen_mask": str(mask_path),
            }
            if getattr(state, "metadata", None):
                if state.metadata.get("token_state_path"):
                    artifact_paths["token_state"] = state.metadata["token_state_path"]
                if state.metadata.get("confidence_path"):
                    artifact_paths["confidence"] = state.metadata["confidence_path"]
            if evaluation.should_abstain:
                stop_reason = "semantic_evaluator_abstained"
                trace_writer.write(make_trace_record(iteration, prompt, current_prompt, evaluation, mask, artifact_paths))
                break
            if not evaluation.has_error:
                stop_reason = "no_semantic_error"
                trace_writer.write(make_trace_record(iteration, prompt, current_prompt, evaluation, mask, artifact_paths))
                break
            if not mask.any():
                stop_reason = "no_actionable_region"
                trace_writer.write(make_trace_record(iteration, prompt, current_prompt, evaluation, mask, artifact_paths))
                break
            current_prompt = compose_correction_prompt(prompt, evaluation)
            correction_prompt_path = artifacts.write_text(f"iterations/{iteration:03d}/correction_prompt.txt", current_prompt)
            artifact_paths["correction_prompt"] = str(correction_prompt_path)
            records.append({
                "iteration": iteration,
                "selected_token_count": mask.count(),
                "evaluation_summary": evaluation.summary,
                "correction_prompt": str(correction_prompt_path),
            })
            trace_writer.write(make_trace_record(iteration, prompt, current_prompt, evaluation, mask, artifact_paths))
            state = self.generator.reopen_and_continue(state, mask, current_prompt, artifacts)
        fallback_applied = False
        if stop_reason == "max_iterations" and self.config.return_initial_on_max_error and initial_decoded_image:
            final_decoded_image = initial_decoded_image
            final_grid_image = initial_grid_image
            fallback_applied = True
        summary = {
            "prompt": prompt,
            "stage1_variant": "direct_token",
            "stop_reason": stop_reason,
            "final_selection_policy": "initial_on_max_error" if self.config.return_initial_on_max_error else "last_candidate",
            "fallback_applied": fallback_applied,
            "iterations_recorded": len(records),
            "evaluator_calls": evaluator_calls,
            "revision_records": records,
            "artifact_root": str(artifacts.root),
            "trace_path": str(artifacts.root / "trace.jsonl"),
            "final_decoded_image": final_decoded_image,
            "final_grid_image": final_grid_image,
            "raw_final_decoded_image": raw_final_decoded_image,
            "raw_final_grid_image": raw_final_grid_image,
            "initial_decoded_image": initial_decoded_image,
            "initial_grid_image": initial_grid_image,
            "started_from_initial_state": initial_state is not None,
        }
        artifacts.write_json("summary.json", summary)
        return summary
