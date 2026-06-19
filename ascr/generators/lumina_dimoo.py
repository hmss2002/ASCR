"""Lumina-DiMOO generator adapter for the ASCR Stage-1 loop.

Mirrors :class:`ascr.generators.mmada.MMaDAAdapter` but drives a
:class:`ascr.generators.lumina_native.LuminaNativeEngine`.  The baseline VQ codes
are kept in ``state.metadata['vq_ids']`` so that ``reopen_and_continue`` can set the
selected token cells back to MASK and re-diffuse only those positions (Lumina's
native masked-diffusion inpainting), with no VQ encode/decode round-trip.
"""

from pathlib import Path

from ascr.core.state import GenerationState
from ascr.generators.base import GeneratorAdapter
from ascr.generators.lumina_native import LuminaNativeEngine


def _flat_to_grid(flat, grid_size):
    return [list(flat[row * grid_size:(row + 1) * grid_size]) for row in range(grid_size)]


class LuminaAdapter(GeneratorAdapter):
    def __init__(
        self,
        checkpoint_path=None,
        repo_path=None,
        lora_path=None,
        device="cuda",
        token_grid_size=64,
        image_size=1024,
        guidance_scale=4.0,
        generation_timesteps=64,
        temperature=1.0,
        seed=1234,
        answer_steps=64,
        answer_block_length=128,
        answer_temperature=0.0,
        answer_cfg_scale=0.0,
    ):
        self.checkpoint_path = checkpoint_path or "models/lumina-dimoo"
        self.repo_path = repo_path
        self.lora_path = lora_path
        self.device = device
        self.token_grid_size = int(token_grid_size)
        self.image_size = int(image_size)
        self.guidance_scale = float(guidance_scale)
        self.generation_timesteps = int(generation_timesteps)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.answer_steps = int(answer_steps)
        self.answer_block_length = int(answer_block_length)
        self.answer_temperature = float(answer_temperature)
        self.answer_cfg_scale = float(answer_cfg_scale)
        self._engine = None

    def engine(self):
        if self._engine is None:
            self._engine = LuminaNativeEngine(
                checkpoint_path=self.checkpoint_path,
                repo_path=self.repo_path,
                lora_path=self.lora_path,
                device=self.device,
                image_size=self.image_size,
                token_grid_size=self.token_grid_size,
                guidance_scale=self.guidance_scale,
                generation_timesteps=self.generation_timesteps,
                temperature=self.temperature,
                answer_steps=self.answer_steps,
                answer_block_length=self.answer_block_length,
                answer_temperature=self.answer_temperature,
                answer_cfg_scale=self.answer_cfg_scale,
            )
        return self._engine

    def initialize(self, prompt, artifacts):
        vq_ids = self.engine().generate(prompt, seed=self.seed)
        return self._state(prompt, 0, vq_ids, {"source": "native_initial"})

    def decode(self, state, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vq_ids = state.metadata.get("vq_ids") if state.metadata else None
        img = self.engine().decode_to(vq_ids, output_path)
        img.convert("RGB").save(output_path)
        state.image_path = str(output_path)
        return state

    def reopen_and_continue(self, state, mask, correction_prompt, artifacts):
        baseline_vq_ids = state.metadata["vq_ids"]
        indices = mask.selected_indices()
        artifacts.write_json(f"iterations/{state.iteration:03d}/semantic_force_mask.json", mask.to_dict())
        vq_ids = self.engine().reopen(
            baseline_vq_ids, indices, correction_prompt, seed=self.seed + state.iteration + 1,
        )
        metadata = {
            "source": "native_reopen_continue",
            "semantic_reopened_tokens": mask.count(),
        }
        return self._state(correction_prompt, state.iteration + 1, vq_ids, metadata)

    def _state(self, prompt, iteration, vq_ids, extra_metadata=None):
        metadata = {"generator": "lumina", "seed": self.seed, "vq_ids": list(vq_ids)}
        if extra_metadata:
            metadata.update(extra_metadata)
        return GenerationState(
            prompt=prompt,
            iteration=iteration,
            token_grid=_flat_to_grid(vq_ids, self.token_grid_size),
            metadata=metadata,
        )
