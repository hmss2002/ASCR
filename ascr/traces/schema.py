TRACE_SCHEMA_VERSION = "stage1.trace.v1"


def make_trace_record(iteration, original_prompt, current_prompt, evaluation, mask, artifact_paths):
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "iteration": iteration,
        "original_prompt": original_prompt,
        "current_prompt": current_prompt,
        "evaluation": evaluation.to_dict(),
        "reopen_mask": mask.to_dict(),
        "artifact_paths": artifact_paths,
        "reserved_for_stage2": {
            "hidden_state_path": None,
            "confidence_map_path": None,
            "revision_gain": None,
            "human_label": None,
        },
    }
