import json
from pathlib import Path

from ascr.core.state import GenerationState
from ascr.generators.base import GeneratorAdapter
from ascr.generators.mmada_native import MMaDANativeEngine
from ascr.generators.showo_native import compact_token_payload, flat_to_grid


class MMaDAAdapter(GeneratorAdapter):
    """MMaDA-8B generator adapter (masked discrete diffusion, native token loop).

    Drives an :class:`MMaDANativeEngine` through the same generator interface used across ASCR.
    The same engine instance can be shared with
    :class:`ascr.evaluators.mmada_self.MMaDASelfEvaluator` via
    :meth:`share_engine_from` so the 8B model is loaded exactly once and both
    generation and self-evaluation run on the same weights.
    """

    def __init__(self, repo_path=None, checkpoint_path=None, vq_model_path=None, device="cuda", token_grid_size=32, image_size=512, guidance_scale=3.5, generation_timesteps=15, seed=1234, max_seq_length=512):
        self.project_root = Path.cwd()
        self.repo_path = Path(repo_path or "external/MMaDA")
        self.checkpoint_path = Path(checkpoint_path or "models/mmada-8b-mixcot")
        self.vq_model_path = Path(vq_model_path or "models/magvitv2")
        self.device = device
        self.token_grid_size = int(token_grid_size)
        self.image_size = int(image_size)
        self.guidance_scale = float(guidance_scale)
        self.generation_timesteps = int(generation_timesteps)
        self.seed = int(seed)
        self.max_seq_length = int(max_seq_length)
        self._native_engine = None

    def initialize(self, prompt, artifacts):
        payload = self._engine().run_confidence_block(prompt, model_tokens=None, steps=self.generation_timesteps, seed=self.seed)
        output_path = artifacts.root / "generator" / "initial_native.png"
        self._engine().decode_tokens(payload["decoded_tokens"], output_path)
        return self._state_from_payload(prompt, 0, payload, output_path, {"source": "native_initial"})

    def decode(self, state, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        decoded_tokens = state.metadata.get("decoded_tokens") if state.metadata else None
        if decoded_tokens:
            self._engine().decode_tokens(decoded_tokens, output_path)
            state.image_path = str(output_path)
            self._write_native_debug_files(state, output_path.parent)
            return state
        from PIL import Image
        Image.open(state.image_path).convert("RGB").save(output_path)
        state.image_path = str(output_path)
        return state

    def reopen_and_continue(self, state, mask, correction_prompt, artifacts):
        model_tokens = state.metadata.get("decoded_tokens") if state.metadata else None
        if not model_tokens:
            model_tokens = self._engine().encode_image(state.image_path)
        semantic_model_tokens = self._engine().force_mask(model_tokens, mask)
        semantic_mask_path = artifacts.write_json(f"iterations/{state.iteration:03d}/semantic_force_mask.json", mask.to_dict())
        payload = self._engine().run_confidence_block(
            correction_prompt,
            model_tokens=semantic_model_tokens,
            steps=self.generation_timesteps,
            seed=self.seed + state.iteration + 1,
        )
        output_path = artifacts.root / "generator" / f"revision_native_{state.iteration + 1:03d}.png"
        self._engine().decode_tokens(payload["decoded_tokens"], output_path)
        metadata = {
            "source": "native_reopen_continue",
            "semantic_reopened_tokens": mask.count(),
            "semantic_mask_path": str(semantic_mask_path),
        }
        return self._state_from_payload(correction_prompt, state.iteration + 1, payload, output_path, metadata)

    def _engine(self):
        if self._native_engine is None:
            self._native_engine = MMaDANativeEngine(
                repo_path=self.repo_path,
                checkpoint_path=self.checkpoint_path,
                vq_model_path=self.vq_model_path,
                device=self.device,
                image_size=self.image_size,
                token_grid_size=self.token_grid_size,
                guidance_scale=self.guidance_scale,
                generation_timesteps=self.generation_timesteps,
                max_seq_length=self.max_seq_length,
            )
        return self._native_engine

    def share_engine_from(self, other):
        if not isinstance(other, MMaDAAdapter):
            return False
        same_engine_config = (
            self.repo_path == other.repo_path
            and self.checkpoint_path == other.checkpoint_path
            and self.vq_model_path == other.vq_model_path
            and self.device == other.device
            and self.image_size == other.image_size
            and self.token_grid_size == other.token_grid_size
            and self.guidance_scale == other.guidance_scale
            and self.generation_timesteps == other.generation_timesteps
            and self.max_seq_length == other.max_seq_length
        )
        if not same_engine_config:
            return False
        self._native_engine = other._engine()
        return True

    def _state_from_payload(self, prompt, iteration, payload, image_path, extra_metadata=None):
        metadata = {
            "generator": "mmada",
            "native_token_loop": True,
            "seed": self.seed,
            "model_tokens": payload["model_tokens"],
            "decoded_tokens": payload["decoded_tokens"],
            "last_confidence": payload["confidence"],
            "last_confidence_mask": payload["confidence_mask"],
            "confidence_steps": payload["confidence_steps"],
            "confidence_step_records": payload["step_records"],
            "mask_token_id": payload["mask_token_id"],
            "confidence_remask_count": sum(1 for value in payload["confidence_mask"] if value),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return GenerationState(
            prompt=prompt,
            iteration=iteration,
            token_grid=flat_to_grid(payload["decoded_tokens"], self.token_grid_size),
            image_path=str(image_path),
            metadata=metadata,
        )

    def _write_native_debug_files(self, state, output_dir):
        if not state.metadata:
            return
        token_path = Path(output_dir) / "token_state.json"
        confidence_path = Path(output_dir) / "confidence.json"
        token_path.write_text(json.dumps(compact_token_payload(state.metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        confidence_payload = {
            "confidence": state.metadata.get("last_confidence", []),
            "confidence_mask": state.metadata.get("last_confidence_mask", []),
            "confidence_remask_count": state.metadata.get("confidence_remask_count", 0),
            "confidence_step_records": state.metadata.get("confidence_step_records", []),
        }
        confidence_path.write_text(json.dumps(confidence_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        state.metadata["token_state_path"] = str(token_path)
        state.metadata["confidence_path"] = str(confidence_path)
