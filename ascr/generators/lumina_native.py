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


class LuminaNativeEngine:
    def __init__(
        self,
        checkpoint_path="models/lumina-dimoo",
        repo_path=None,
        device="cuda",
        image_size=1024,
        token_grid_size=64,
        guidance_scale=4.0,
        generation_timesteps=64,
        temperature=1.0,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.repo_path = str(repo_path or os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO"))
        self.device = device
        self.image_size = int(image_size)
        self.token_grid_size = int(token_grid_size)
        self.guidance_scale = float(guidance_scale)
        self.generation_timesteps = int(generation_timesteps)
        self.temperature = float(temperature)
        self._model = None
        self._tokenizer = None
        self._vqvae = None
        self._lumina = None
        self._torch = None

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
        self._model.eval()
        self._vqvae = VQModel.from_pretrained(self.checkpoint_path, subfolder="vqvae").to(self.device)

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
        return self._sample(prompt, img_token_list, seed)

    def reopen(self, baseline_vq_ids, selected_indices, prompt, seed=0):
        self._load()
        h, w = self._grid_hw()
        tokens = list(baseline_vq_ids)
        for row, col in selected_indices:
            if 0 <= row < h and 0 <= col < w:
                tokens[row * w + col] = MASK_TOKEN_ID
        return self._sample(prompt, tokens, seed)

    def decode_to(self, vq_ids, output_path):
        self._load()
        iu = self._lumina["image_utils"]
        device = next(self._model.parameters()).device
        codes = self._torch.tensor(vq_ids, device=device, dtype=self._torch.long).view(1, -1)
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
    def answer_image(self, question, image_path, max_new_tokens=384):
        """Image-conditioned text generation via Lumina native MMU pipeline.

        Uses the official Lumina-DiMOO generate_text_understanding path:
        encode image -> build MMU prompt -> masked diffusion text generation.
        """
        self._load()
        iu = self._lumina["image_utils"]
        pu = self._lumina["prompt_utils"]
        from generators.text_understanding_generator import generate_text_understanding

        # --- encode image -------------------------------------------------
        from PIL import Image
        img = Image.open(str(image_path)).convert("RGB")
        crop_size_list = iu.generate_crop_size_list((self.image_size // 32) ** 2, 32)
        img = iu.var_center_crop(img, crop_size_list=crop_size_list)
        iw, ih = img.size
        vae_scale = 2 ** (len(self._vqvae.config.block_out_channels) - 1)
        seq_len, newline_every, token_grid_h, token_grid_w = iu.calculate_vq_params(ih, iw, vae_scale)
        img_tokens = iu.encode_img_with_breaks(img, vqvae=self._vqvae)
        img_tokens = iu.add_break_line(img_tokens, token_grid_h, token_grid_w, new_number=NEWLINE_TOKEN_ID)

        # --- build prompt -------------------------------------------------
        input_prompt = pu.generate_multimodal_understanding_prompt(question)
        input_ids = self._tokenizer(input_prompt)["input_ids"]
        input_token = input_ids[:-1] + img_tokens + input_ids[-1:]
        code_start = len(input_token) + 1
        input_token = input_token + [ANSWER_START] + [MASK_TOKEN_ID] * int(max_new_tokens) + [ANSWER_END]
        device = next(self._model.parameters()).device
        input_ids_t = self._torch.tensor(input_token, device=device).unsqueeze(0)

        # --- generate text ------------------------------------------------
        out = generate_text_understanding(
            self._model, input_ids_t,
            steps=128, gen_length=int(max_new_tokens), block_length=min(256, int(max_new_tokens)),
            temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
            code_start=code_start,
        )
        text = self._tokenizer.batch_decode(out[:, code_start:-1], skip_special_tokens=True)[0]
        return text.strip()

