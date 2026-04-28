import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from ascr.core.state import GenerationState
from ascr.generators.base import GeneratorAdapter
from ascr.generators.showo_native import ShowONativeEngine, compact_token_payload, flat_to_grid


class ShowOAdapter(GeneratorAdapter):
    def __init__(self, repo_path=None, checkpoint_path=None, vq_model_path=None, llm_model_path=None, showo_config_path=None, device="cuda", token_grid_size=32, image_size=512, guidance_scale=4.0, generation_timesteps=18, seed=1234, native_token_loop=True, confidence_steps=2):
        self.project_root = Path.cwd()
        self.repo_path = Path(repo_path or "external/Show-o")
        self.checkpoint_path = Path(checkpoint_path or "models/show-o-512x512")
        self.vq_model_path = Path(vq_model_path or "models/magvitv2")
        self.llm_model_path = Path(llm_model_path or "models/phi-1_5")
        self.showo_config_path = Path(showo_config_path or "configs/showo_local_512x512.yaml")
        self.device = device
        self.token_grid_size = int(token_grid_size)
        self.image_size = int(image_size)
        self.guidance_scale = float(guidance_scale)
        self.generation_timesteps = int(generation_timesteps)
        self.seed = int(seed)
        self.native_token_loop = bool(native_token_loop)
        self.confidence_steps = int(confidence_steps)
        self._native_engine = None

    def initialize(self, prompt, artifacts):
        if self.native_token_loop:
            payload = self._engine().run_confidence_block(prompt, model_tokens=None, steps=self.confidence_steps, seed=self.seed)
            output_path = artifacts.root / "generator" / "initial_native.png"
            self._engine().decode_tokens(payload["decoded_tokens"], output_path)
            return self._state_from_payload(prompt, 0, payload, output_path, {"source": "native_initial"})
        output_path = artifacts.root / "generator" / "initial.png"
        self.generate_t2i(prompt, output_path, seed=self.seed)
        token_grid = [[0 for _ in range(self.token_grid_size)] for _ in range(self.token_grid_size)]
        return GenerationState(prompt=prompt, iteration=0, token_grid=token_grid, image_path=str(output_path), metadata={"generator": "showo", "seed": self.seed, "native_token_loop": False})

    def decode(self, state, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        decoded_tokens = state.metadata.get("decoded_tokens") if state.metadata else None
        if self.native_token_loop and decoded_tokens:
            self._engine().decode_tokens(decoded_tokens, output_path)
            state.image_path = str(output_path)
            self._write_native_debug_files(state, output_path.parent)
            return state
        Image.open(state.image_path).convert("RGB").save(output_path)
        state.image_path = str(output_path)
        return state

    def reopen_and_continue(self, state, mask, correction_prompt, artifacts):
        if self.native_token_loop:
            model_tokens = state.metadata.get("model_tokens") if state.metadata else None
            if not model_tokens:
                model_tokens = self._engine().encode_image(state.image_path)
            semantic_model_tokens = self._engine().force_mask(model_tokens, mask)
            semantic_mask_path = artifacts.write_json(f"iterations/{state.iteration:03d}/semantic_force_mask.json", mask.to_dict())
            payload = self._engine().run_confidence_block(
                correction_prompt,
                model_tokens=semantic_model_tokens,
                steps=self.confidence_steps,
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
        mask_path = artifacts.iteration_dir(state.iteration) / "inpainting_mask.png"
        self.write_inpainting_mask(mask, mask_path)
        output_path = artifacts.root / "generator" / f"revision_{state.iteration + 1:03d}.png"
        self.generate_inpainting(correction_prompt, state.image_path, mask_path, output_path, seed=self.seed + state.iteration + 1)
        next_grid = [row[:] for row in state.token_grid]
        for row, col in mask.selected_indices():
            next_grid[row][col] = 1
        return GenerationState(prompt=correction_prompt, iteration=state.iteration + 1, token_grid=next_grid, image_path=str(output_path), metadata={"generator": "showo", "native_token_loop": False, "reopened_tokens": mask.count(), "mask_path": str(mask_path)})

    def generate_t2i(self, prompt, output_path, seed=None):
        return self._run_script("scripts/run_showo_t2i_local.sh", prompt, output_path, seed=seed)

    def generate_inpainting(self, prompt, input_image, mask_image, output_path, seed=None):
        return self._run_script("scripts/run_showo_inpaint_local.sh", prompt, output_path, seed=seed, extra_env={"INPUT_IMAGE": str(input_image), "MASK_IMAGE": str(mask_image)})

    def write_inpainting_mask(self, mask, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("L", (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(image)
        step = self.image_size / float(mask.token_grid_size)
        for row, col in mask.selected_indices():
            left = int(col * step)
            top = int(row * step)
            right = int((col + 1) * step)
            bottom = int((row + 1) * step)
            draw.rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), fill=255)
        image.save(output_path)
        return output_path

    def _engine(self):
        if self._native_engine is None:
            self._native_engine = ShowONativeEngine(
                repo_path=self.repo_path,
                checkpoint_path=self.checkpoint_path,
                vq_model_path=self.vq_model_path,
                llm_model_path=self.llm_model_path,
                showo_config_path=self.showo_config_path,
                device=self.device,
                image_size=self.image_size,
                token_grid_size=self.token_grid_size,
                guidance_scale=self.guidance_scale,
                generation_timesteps=self.generation_timesteps,
            )
        return self._native_engine

    def _state_from_payload(self, prompt, iteration, payload, image_path, extra_metadata=None):
        metadata = {
            "generator": "showo",
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

    def _run_script(self, script, prompt, output_path, seed=None, extra_env=None):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "SHOWO_REPO_DIR": str(self.repo_path),
            "SHOWO_MODEL_ROOT": str(self._model_root()),
            "SHOWO_CONFIG": str(self.showo_config_path),
            "OUTPUT_IMAGE": str(output_path),
            "BATCH_SIZE": "1",
            "GUIDANCE_SCALE": str(self.guidance_scale),
            "GENERATION_TIMESTEPS": str(self.generation_timesteps),
            "ASCR_SEED": str(seed or self.seed),
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(self.project_root / ".hf_home"),
            "HUGGINGFACE_HUB_CACHE": str(self.project_root / ".hf_home" / "hub"),
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        })
        if extra_env:
            env.update(extra_env)
        command = ["bash", str(self.project_root / script), prompt]
        completed = subprocess.run(command, cwd=str(self.project_root), env=env, capture_output=True, text=True)
        log_path = output_path.with_suffix(output_path.suffix + ".log")
        log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Show-o script failed; see {log_path}")
        if not output_path.exists():
            raise RuntimeError(f"Show-o script did not create {output_path}; see {log_path}")
        return output_path

    def _model_root(self):
        if self.checkpoint_path.name == "show-o-512x512":
            return self.checkpoint_path.parent
        return Path("models")
