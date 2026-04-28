from dataclasses import dataclass

from ascr.core.artifacts import RunArtifacts, current_git_commit
from ascr.grids.overlay import create_grid_overlay
from ascr.revision.prompt_composer import compose_correction_prompt
from ascr.traces.schema import make_trace_record
from ascr.traces.writer import TraceWriter


@dataclass
class ASCRRunConfig:
    run_name: str = "stage1"
    max_iterations: int = 3
    image_size: int = 256
    coarse_grid_size: int = 4
    token_grid_size: int = 16
    output_dir: str = "outputs/stage1"


class ASCRLoop:
    def __init__(self, generator, evaluator, selector, config):
        self.generator = generator
        self.evaluator = evaluator
        self.selector = selector
        self.config = config

    def run(self, prompt, project_root=".", initial_state=None):
        artifacts = RunArtifacts.create(self.config.output_dir, self.config.run_name)
        artifacts.write_json("config_snapshot.json", {
            "run_name": self.config.run_name,
            "max_iterations": self.config.max_iterations,
            "image_size": self.config.image_size,
            "coarse_grid_size": self.config.coarse_grid_size,
            "token_grid_size": self.config.token_grid_size,
            "git_commit": current_git_commit(project_root),
            "started_from_initial_state": initial_state is not None,
        })
        trace_writer = TraceWriter(artifacts.root / "trace.jsonl")
        state = initial_state if initial_state is not None else self.generator.initialize(prompt, artifacts)
        records = []
        stop_reason = "max_iterations"
        current_prompt = prompt
        final_decoded_image = None
        final_grid_image = None
        for iteration in range(self.config.max_iterations):
            iteration_dir = artifacts.iteration_dir(iteration)
            decoded_path = iteration_dir / "decoded.ppm"
            state = self.generator.decode(state, decoded_path)
            final_decoded_image = str(decoded_path)
            grid_path = iteration_dir / "grid.ppm"
            create_grid_overlay(decoded_path, grid_path, image_size=self.config.image_size, grid_size=self.config.coarse_grid_size)
            final_grid_image = str(grid_path)
            evaluation = self.evaluator.evaluate(prompt, str(grid_path), iteration, current_prompt=current_prompt)
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
            trace_writer.write(make_trace_record(iteration, prompt, current_prompt, evaluation, mask, artifact_paths))
            if evaluation.should_abstain:
                stop_reason = "semantic_evaluator_abstained"
                break
            if not evaluation.has_error:
                stop_reason = "no_semantic_error"
                break
            if not mask.any():
                stop_reason = "no_actionable_region"
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
            state = self.generator.reopen_and_continue(state, mask, current_prompt, artifacts)
        summary = {
            "prompt": prompt,
            "stop_reason": stop_reason,
            "iterations_recorded": len(records),
            "artifact_root": str(artifacts.root),
            "trace_path": str(artifacts.root / "trace.jsonl"),
            "final_decoded_image": final_decoded_image,
            "final_grid_image": final_grid_image,
            "started_from_initial_state": initial_state is not None,
        }
        artifacts.write_json("summary.json", summary)
        return summary


def run_config_from_mapping(mapping):
    return ASCRRunConfig(
        run_name=str(mapping.get("run_name", "stage1")),
        max_iterations=int(mapping.get("max_iterations", 3)),
        image_size=int(mapping.get("image_size", 256)),
        coarse_grid_size=int(mapping.get("coarse_grid_size", 4)),
        token_grid_size=int(mapping.get("token_grid_size", 16)),
        output_dir=str(mapping.get("output_dir", "outputs/stage1")),
    )
