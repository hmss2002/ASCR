"""Native Lumina-DiMOO engine for ASCR Stage-1.

Wraps the vendored ``Alpha-VLLM/Lumina-DiMOO`` text-to-image discrete-diffusion
pipeline so the 8B model + VQ-VAE are loaded exactly once per process and exposes
two operations ASCR needs:

* ``generate(prompt)``  -> full masked-diffusion sampling (baseline image).
* ``reopen(vq_ids, indices, prompt)`` -> re-diffuse ONLY the selected token cells
  by setting them back to the MASK token while keeping every other baseline code
  fixed.  This is Lumina's native inpainting path, used here at token granularity.

The Lumina repository is imported lazily from ``LUMINA_REPO`` (default
``third_party/Lumina-DiMOO``) so importing this module never pulls heavy deps into
the base test environment.
"""

import os
import sys
from pathlib import Path

from ascr.core.peft_compat import ensure_transformers_tensor_parallel_compat


# Lumina special-token ids (from Lumina-DiMOO/config.py SPECIAL_TOKENS).
MASK_TOKEN_ID = 126336
NEWLINE_TOKEN_ID = 126084
IMAGE_TOKEN_OFFSET = 126356
ANSWER_START = 126354
ANSWER_END = 126355
BOI = 126349
EOI = 126350
CODEBOOK_SIZE = 8192
VAE_SCALE = 16


def align_answer_generation_lengths(max_new_tokens, block_length, steps):
    """Return Lumina text-generation lengths that satisfy block constraints."""
    block_len = max(1, int(block_length))
    gen_len = max(1, int(max_new_tokens))
    gen_len = (gen_len // block_len) * block_len
    if gen_len < block_len:
        gen_len = block_len
    num_blocks = max(1, gen_len // block_len)
    aligned_steps = max(1, int(steps))
    if aligned_steps % num_blocks != 0:
        aligned_steps = (aligned_steps // num_blocks) * num_blocks
        if aligned_steps < num_blocks:
            aligned_steps = num_blocks
    return gen_len, block_len, aligned_steps


class LuminaNativeEngine:
    def __init__(
        self,
        checkpoint_path="models/lumina-dimoo",
        repo_path=None,
        lora_path=None,
        device="cuda",
        image_size=1024,
        token_grid_size=64,
        guidance_scale=4.0,
        generation_timesteps=64,
        temperature=1.0,
        answer_steps=64,
        answer_block_length=128,
        answer_temperature=0.0,
        answer_cfg_scale=0.0,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.repo_path = str(repo_path or os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO"))
        self.lora_path = str(lora_path) if lora_path else None
        self.device = device
        self.image_size = int(image_size)
        self.token_grid_size = int(token_grid_size)
        self.guidance_scale = float(guidance_scale)
        self.generation_timesteps = int(generation_timesteps)
        self.temperature = float(temperature)
        self.answer_steps = int(answer_steps)
        self.answer_block_length = int(answer_block_length)
        self.answer_temperature = float(answer_temperature)
        self.answer_cfg_scale = float(answer_cfg_scale)
        self._model = None
        self._tokenizer = None
        self._vqvae = None
        self._lumina = None
        self._torch = None

    def unload(self, clear_lora=False):
        """Release loaded Lumina weights so the same engine can reload later."""
        had_loaded_state = any(value is not None for value in (self._model, self._tokenizer, self._vqvae))
        torch = self._torch
        self._model = None
        self._tokenizer = None
        self._vqvae = None
        if clear_lora:
            self.lora_path = None
        import gc

        gc.collect()
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return had_loaded_state

    def release_generation_cache(self):
        """Release transient generation cache while keeping loaded weights resident."""
        had_loaded_state = any(value is not None for value in (self._model, self._tokenizer, self._vqvae))
        torch = self._torch
        import gc

        gc.collect()
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
        return had_loaded_state

    def attach_lora(self, lora_path):
        """Attach an MMU LoRA adapter to the loaded model, or defer until load."""
        if not lora_path:
            return self
        lora_path = str(lora_path)
        if self.lora_path == lora_path:
            return self
        if self._model is None:
            self.lora_path = lora_path
            return self
        ensure_transformers_tensor_parallel_compat()
        from peft import PeftModel

        self._model = PeftModel.from_pretrained(self._model, lora_path)
        self._model.eval()
        self.lora_path = lora_path
        return self

    # ------------------------------------------------------------------ load
    def _ensure_repo_on_path(self):
        repo = str(Path(self.repo_path).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def _load(self):
        if self._model is not None:
            return
        self._ensure_repo_on_path()
        import torch
        from transformers import AutoTokenizer
        from diffusers import VQModel
        from model import LLaDAForMultiModalGeneration
        from utils import image_utils, prompt_utils
        from generators.image_generation_generator import generate_image

        self._torch = torch
        self._lumina = {
            "image_utils": image_utils,
            "prompt_utils": prompt_utils,
            "generate_image": generate_image,
            "templates": prompt_utils.create_prompt_templates(),
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        self._model = LLaDAForMultiModalGeneration.from_pretrained(
            self.checkpoint_path, torch_dtype=self._torch.bfloat16, device_map="auto",
        )
        if self.lora_path:
            ensure_transformers_tensor_parallel_compat()
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
        self._model.eval()
        self._vqvae = VQModel.from_pretrained(self.checkpoint_path, subfolder="vqvae").to(self.device)

    def _inference_context(self):
        if hasattr(self._torch, "inference_mode"):
            return self._torch.inference_mode()
        return self._torch.no_grad()

    # --------------------------------------------------------------- helpers
    def _grid_hw(self):
        h = self.image_size // VAE_SCALE
        w = self.image_size // VAE_SCALE
        return h, w

    def _build_prompt_ids(self, prompt, img_token_list):
        """img_token_list: flat list of length seq_len in OFFSET space (or MASK ids)."""
        iu = self._lumina["image_utils"]
        pu = self._lumina["prompt_utils"]
        h, w = self._grid_hw()
        input_prompt, uncon_prompt = pu.generate_text_to_image_prompt(prompt, self._lumina["templates"])
        con_tokens = self._tokenizer(input_prompt)["input_ids"]
        uncon_tokens = self._tokenizer(uncon_prompt)["input_ids"]
        img_with_breaks = iu.add_break_line(img_token_list, h, w, new_number=NEWLINE_TOKEN_ID)
        img_pred = [ANSWER_START] + [BOI] + img_with_breaks + [EOI] + [ANSWER_END]
        device = next(self._model.parameters()).device
        prompt_ids = self._torch.tensor(con_tokens + img_pred, device=device).unsqueeze(0)
        uncon_ids = self._torch.tensor(uncon_tokens, device=device).unsqueeze(0)
        code_start = len(con_tokens) + 2
        return prompt_ids, uncon_ids, code_start

    def _sample(self, prompt, img_token_list, seed):
        h, w = self._grid_hw()
        seq_len = h * w
        if seed:
            from utils.generation_utils import setup_seed
            setup_seed(int(seed))
        prompt_ids, uncon_ids, code_start = self._build_prompt_ids(prompt, img_token_list)
        vq_tokens = self._lumina["generate_image"](
            self._model,
            prompt_ids,
            seq_len=seq_len,
            newline_every=w,
            timesteps=self.generation_timesteps,
            mask_token_id=MASK_TOKEN_ID,
            newline_id=NEWLINE_TOKEN_ID,
            temperature=self.temperature,
            cfg_scale=self.guidance_scale,
            uncon_ids=uncon_ids,
            code_start=code_start,
            codebook_size=CODEBOOK_SIZE,
            text_vocab_size=IMAGE_TOKEN_OFFSET,
        )
        return vq_tokens.view(-1).tolist()  # flat ids in OFFSET space, length seq_len

    # ----------------------------------------------------------------- public
    def generate(self, prompt, seed=0):
        self._load()
        h, w = self._grid_hw()
        seq_len = h * w
        img_token_list = [MASK_TOKEN_ID] * seq_len
        with self._inference_context():
            return self._sample(prompt, img_token_list, seed)

    def reopen(self, baseline_vq_ids, selected_indices, prompt, seed=0):
        self._load()
        h, w = self._grid_hw()
        tokens = list(baseline_vq_ids)
        for row, col in selected_indices:
            if 0 <= row < h and 0 <= col < w:
                tokens[row * w + col] = MASK_TOKEN_ID
        with self._inference_context():
            return self._sample(prompt, tokens, seed)

    def decode_to(self, vq_ids, output_path):
        self._load()
        iu = self._lumina["image_utils"]
        device = next(self._model.parameters()).device
        codes = self._torch.tensor(vq_ids, device=device, dtype=self._torch.long).view(1, -1)
        with self._inference_context():
            img = iu.decode_vq_to_image(
                codes, str(output_path),
                vae_ckpt=self.checkpoint_path,
                image_height=self.image_size,
                image_width=self.image_size,
                vqvae=self._vqvae,
            )
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(str(output_path))
        return img

    # ----------------------------------------------------------- mmu / answer
    def _normalise_mmu_vq_tokens(self, vq_ids):
        """Return flat image-token ids in Lumina's MMU offset token space."""
        values = [int(value) for value in vq_ids]
        specials = {MASK_TOKEN_ID, NEWLINE_TOKEN_ID, BOI, EOI, ANSWER_START, ANSWER_END}
        non_special = [value for value in values if value not in specials]
        if non_special and all(0 <= value < CODEBOOK_SIZE for value in non_special):
            return [value if value in specials else value + IMAGE_TOKEN_OFFSET for value in values]
        return values

    def _answer_from_mmu_tokens(self, question, img_tokens_with_breaks, max_new_tokens=384):
        """Generate an MMU text answer from already prepared image tokens."""
        pu = self._lumina["prompt_utils"]
        from generators.text_understanding_generator import generate_text_understanding

        gen_len, block_len, steps = align_answer_generation_lengths(
            max_new_tokens,
            self.answer_block_length,
            self.answer_steps,
        )

        input_prompt = pu.generate_multimodal_understanding_prompt(question)
        input_ids = self._tokenizer(input_prompt)["input_ids"]
        input_token = input_ids[:-1] + list(img_tokens_with_breaks) + input_ids[-1:]
        code_start = len(input_token) + 1
        input_token = input_token + [ANSWER_START] + [MASK_TOKEN_ID] * gen_len + [ANSWER_END]
        device = next(self._model.parameters()).device
        input_ids_t = self._torch.tensor(input_token, device=device).unsqueeze(0)

        out = generate_text_understanding(
            self._model, input_ids_t,
            steps=steps, gen_length=gen_len, block_length=block_len,
            temperature=self.answer_temperature, cfg_scale=self.answer_cfg_scale, remasking="low_confidence",
            code_start=code_start,
        )
        text = self._tokenizer.batch_decode(out[:, code_start:-1], skip_special_tokens=True)[0]
        return text.strip()

    def answer_image(self, question, image_path, max_new_tokens=384):
        """Image-conditioned text generation via Lumina native MMU pipeline.

        Uses the official Lumina-DiMOO generate_text_understanding path:
        encode image -> build MMU prompt -> masked diffusion text generation.
        """
        self._load()
        iu = self._lumina["image_utils"]

        # --- encode image -------------------------------------------------
        from PIL import Image
        with Image.open(str(image_path)) as opened:
            img = opened.convert("RGB")
        crop_size_list = iu.generate_crop_size_list((self.image_size // 32) ** 2, 32)
        img = iu.var_center_crop(img, crop_size_list=crop_size_list)
        iw, ih = img.size
        vae_scale = 2 ** (len(self._vqvae.config.block_out_channels) - 1)
        _seq_len, _newline_every, token_grid_h, token_grid_w = iu.calculate_vq_params(ih, iw, vae_scale)
        with self._inference_context():
            img_tokens = iu.encode_img_with_breaks(img, vqvae=self._vqvae)
            img_tokens = iu.add_break_line(img_tokens, token_grid_h, token_grid_w, new_number=NEWLINE_TOKEN_ID)

            return self._answer_from_mmu_tokens(question, img_tokens, max_new_tokens=max_new_tokens)

    def answer_vq_tokens(self, question, vq_ids, max_new_tokens=384):
        """MMU text generation from existing Lumina VQ tokens.

        Stage-4 self-corruption already has corrupted VQ ids. This method lets
        the MMU path consume that internal representation directly instead of
        decoding to an RGB image and re-encoding it through the VQ-VAE.
        """
        self._load()
        iu = self._lumina["image_utils"]
        h, w = self._grid_hw()
        flat_tokens = self._normalise_mmu_vq_tokens(vq_ids)
        expected = h * w
        if len(flat_tokens) != expected:
            raise ValueError(f"Expected {expected} VQ ids for {h}x{w}, got {len(flat_tokens)}")
        img_tokens = iu.add_break_line(flat_tokens, h, w, new_number=NEWLINE_TOKEN_ID)
        with self._inference_context():
            return self._answer_from_mmu_tokens(question, img_tokens, max_new_tokens=max_new_tokens)
