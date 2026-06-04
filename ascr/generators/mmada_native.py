import os
import sys
from pathlib import Path


class MMaDANativeEngine:
    """Single-load MMaDA-8B engine shared by the generator and the self-evaluator.

    Mirrors the method surface of :class:`ascr.generators.showo_native.ShowONativeEngine`
    (``run_confidence_block`` / ``encode_image`` / ``decode_tokens`` / ``force_mask`` /
    ``answer_image``) so the existing ``DirectTokenReopenLoop`` can drive it unchanged.

    The same loaded MMaDA-8B model serves two roles:
      * generation: masked discrete diffusion (``MMadaModelLM.t2i_generate``);
      * self-evaluation: image-to-text understanding (``MMadaModelLM.mmu_generate``).

    Because MMaDA uses the same MAGVIT-v2 tokenizer (codebook 8192, 32x32=1024 image
    tokens at 512px) as Show-o, the token-grid / force-mask / reopen contract is
    identical, so ``model_tokens`` keep Show-o's convention: masked positions hold the
    model's ``mask_token_id`` and known positions hold the raw VQ id (0..codebook-1).
    """

    def __init__(self, repo_path="external/MMaDA", checkpoint_path="models/mmada-8b-mixcot", vq_model_path="models/magvitv2", device="cuda", image_size=512, token_grid_size=32, guidance_scale=3.5, generation_timesteps=15, max_seq_length=512):
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.vq_model_path = Path(vq_model_path)
        self.device_name = device
        self.image_size = int(image_size)
        self.token_grid_size = int(token_grid_size)
        self.guidance_scale = float(guidance_scale)
        self.generation_timesteps = int(generation_timesteps)
        self.max_seq_length = int(max_seq_length)
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        repo = str(self.repo_path.resolve())
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from PIL import Image
        from transformers import AutoTokenizer
        from models import MAGVITv2, get_mask_schedule, MMadaModelLM
        from training.prompting_utils import UniversalPrompting
        from training.utils import image_transform
        self.torch = torch
        self.Image = Image
        self.image_transform = image_transform
        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        checkpoint = str(self.checkpoint_path.resolve())
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left", local_files_only=True)
        self.uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=self.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100,
            cond_dropout_prob=0.1,
            use_reserved_token=True,
        )
        self.vq_model = MAGVITv2.from_pretrained(str(self.vq_model_path.resolve()), local_files_only=True).to(self.device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()
        self.model = MMadaModelLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True).to(self.device)
        self.model.eval()
        self.mask_schedule = get_mask_schedule("cosine")
        self.num_vq_tokens = self.token_grid_size * self.token_grid_size
        self.codebook_size = 8192
        self.token_offset = len(self.uni_prompting.text_tokenizer)
        self.mask_token_id = int(self.model.config.mask_token_id)
        self._loaded = True
        return self

    def empty_model_tokens(self):
        self.load()
        return [self.mask_token_id for _ in range(self.num_vq_tokens)]

    def encode_image(self, image_path):
        self.load()
        image = self.Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image, resolution=self.image_size).to(self.device).unsqueeze(0)
        with self.torch.no_grad():
            tokens = self.vq_model.get_code(image_tensor)
        return tokens.reshape(-1).detach().cpu().long().tolist()

    def decode_tokens(self, decoded_tokens, output_path):
        self.load()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokens = self.torch.tensor(decoded_tokens, dtype=self.torch.long, device=self.device).reshape(1, -1)
        tokens = self.torch.clamp(tokens, min=0, max=self.codebook_size - 1)
        with self.torch.no_grad():
            images = self.vq_model.decode_code(tokens)
        images = self.torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        image = images[0].permute(1, 2, 0).detach().cpu().float().numpy()
        image = (image * 255.0).round().clip(0, 255).astype("uint8")
        self.Image.fromarray(image).save(output_path)
        return output_path

    def _image_prompt_tokens(self, model_tokens):
        self.load()
        tokens = self.torch.tensor(model_tokens, dtype=self.torch.long, device=self.device).reshape(1, -1)
        return self.torch.where(tokens == self.mask_token_id, tokens, tokens + self.token_offset)

    def run_confidence_block(self, prompt, model_tokens=None, steps=None, seed=None):
        self.load()
        if model_tokens is None:
            model_tokens = self.empty_model_tokens()
        steps = int(steps or self.generation_timesteps)
        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        image_prompt_tokens = self._image_prompt_tokens(model_tokens)
        input_ids, attention_mask = self.uni_prompting(([prompt], image_prompt_tokens), "t2i_gen")
        if self.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = self.uni_prompting(([""], image_prompt_tokens), "t2i_gen")
        else:
            uncond_input_ids = None
            uncond_attention_mask = None
        with self.torch.no_grad():
            sampled_ids = self.model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=self.guidance_scale,
                temperature=1.0,
                timesteps=steps,
                noise_schedule=self.mask_schedule,
                generator=generator,
                config=None,
                seq_len=self.num_vq_tokens,
                mask_token_id=self.mask_token_id,
                resolution=self.max_seq_length,
                codebook_size=self.codebook_size,
                uni_prompting=self.uni_prompting,
            )
        decoded_tokens = self.torch.clamp(sampled_ids.reshape(-1), min=0, max=self.codebook_size - 1).detach().cpu().long().tolist()
        return {
            "model_tokens": list(decoded_tokens),
            "decoded_tokens": list(decoded_tokens),
            "confidence": [],
            "confidence_mask": [],
            "confidence_steps": steps,
            "step_records": [],
            "mask_token_id": self.mask_token_id,
        }

    def force_mask(self, model_tokens, token_mask):
        self.load()
        next_tokens = list(model_tokens)
        for row, col in token_mask.selected_indices():
            index = int(row) * self.token_grid_size + int(col)
            if 0 <= index < len(next_tokens):
                next_tokens[index] = self.mask_token_id
        return next_tokens

    def token_confidence(self, prompt, model_tokens):
        """Score the model's confidence in each of its own image tokens.

        Runs a single MMaDA forward pass over the *current* (fully known) image
        tokens in the text-to-image layout and reads, for every one of the
        ``num_vq_tokens`` image positions, the softmax probability the model
        assigns to the token that is actually present. Lower probability means the
        model is less confident that token is correct, i.e. a stronger candidate
        to reopen. This lets MMaDA judge its own 1024 discrete tokens directly,
        with no down-sampling and no external selector.

        Returns a list of ``num_vq_tokens`` floats in ``[0, 1]`` (row-major).
        """
        self.load()
        if model_tokens is None:
            model_tokens = self.empty_model_tokens()
        image_prompt_tokens = self._image_prompt_tokens(model_tokens)
        input_ids, attention_mask = self.uni_prompting(([prompt], image_prompt_tokens), "t2i_gen")
        offset = self.token_offset
        num_vq = self.num_vq_tokens
        with self.torch.no_grad():
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            logits = self.model(input_ids, attention_bias=attention_bias).logits
            image_logits = logits[:, -(num_vq + 1):-1, offset:offset + self.codebook_size]
            probs = image_logits.softmax(dim=-1)
            present = self.torch.tensor(model_tokens, dtype=self.torch.long, device=probs.device).reshape(1, -1)
            present = self.torch.clamp(present, min=0, max=self.codebook_size - 1)
            gathered = self.torch.gather(probs, -1, present.unsqueeze(-1)).squeeze(-1)
        return gathered.reshape(-1).detach().cpu().float().tolist()

    def answer_image(self, question, image_path, max_new_tokens=256):
        self.load()
        image_tokens = self.torch.tensor(self.encode_image(image_path), dtype=self.torch.long, device=self.device).reshape(1, -1)
        image_tokens = image_tokens + self.token_offset
        messages = [{"role": "user", "content": question}]
        text_token_ids = self.uni_prompting.text_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        batch_size = image_tokens.shape[0]
        input_ids = self.torch.cat([
            (self.torch.ones(batch_size, 1, device=self.device) * self.uni_prompting.sptids_dict["<|mmu|>"].to(self.device)).long(),
            (self.torch.ones(batch_size, 1, device=self.device) * self.uni_prompting.sptids_dict["<|soi|>"].to(self.device)).long(),
            image_tokens,
            (self.torch.ones(batch_size, 1, device=self.device) * self.uni_prompting.sptids_dict["<|eoi|>"].to(self.device)).long(),
            text_token_ids,
        ], dim=1).long()
        max_new_tokens = int(max_new_tokens)
        steps = max(1, max_new_tokens // 2)
        block_length = max(1, max_new_tokens // 4)
        with self.torch.no_grad():
            if self.device.type == "cuda":
                with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                    output_ids = self.model.mmu_generate(input_ids, max_new_tokens=max_new_tokens, steps=steps, block_length=block_length)
            else:
                output_ids = self.model.mmu_generate(input_ids, max_new_tokens=max_new_tokens, steps=steps, block_length=block_length)
        generated_ids = output_ids[:, input_ids.shape[1]:]
        return self.uni_prompting.text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
