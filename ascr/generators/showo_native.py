import os
import sys
from pathlib import Path


class ShowONativeEngine:
    def __init__(self, repo_path="external/Show-o", checkpoint_path="models/show-o-512x512", vq_model_path="models/magvitv2", llm_model_path="models/phi-1_5", showo_config_path="configs/showo_local_512x512.yaml", device="cuda", image_size=512, token_grid_size=32, guidance_scale=4.0, generation_timesteps=18):
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.vq_model_path = Path(vq_model_path)
        self.llm_model_path = Path(llm_model_path)
        self.showo_config_path = Path(showo_config_path)
        self.device_name = device
        self.image_size = int(image_size)
        self.token_grid_size = int(token_grid_size)
        self.guidance_scale = float(guidance_scale)
        self.generation_timesteps = int(generation_timesteps)
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
        from omegaconf import OmegaConf
        from PIL import Image
        from transformers import AutoTokenizer
        from models import MAGVITv2, Showo
        from models.sampling import get_mask_chedule, mask_by_random_topk
        from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu, create_attention_mask_predict_next
        from training.utils import image_transform
        self.torch = torch
        self.Image = Image
        self.mask_by_random_topk = mask_by_random_topk
        self.create_attention_mask_predict_next = create_attention_mask_predict_next
        self.create_attention_mask_for_mmu = create_attention_mask_for_mmu
        self.image_transform = image_transform
        config = OmegaConf.load(str(self.showo_config_path))
        config.model.showo.pretrained_model_path = str(self.checkpoint_path.resolve())
        config.model.showo.llm_model_path = str(self.llm_model_path.resolve())
        config.model.vq_model.vq_model_name = str(self.vq_model_path.resolve())
        config.dataset.params.resolution = self.image_size
        config.dataset.preprocessing.resolution = self.image_size
        config.training.batch_size = 1
        config.training.guidance_scale = self.guidance_scale
        config.training.generation_timesteps = self.generation_timesteps
        self.config = config
        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(self.llm_model_path.resolve()), padding_side="left", local_files_only=True)
        self.uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob,
        )
        self.vq_model = MAGVITv2.from_pretrained(str(self.vq_model_path.resolve()), local_files_only=True).to(self.device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()
        self.model = Showo.from_pretrained(str(self.checkpoint_path.resolve()), llm_model_path=str(self.llm_model_path.resolve()), local_files_only=True).to(self.device)
        self.model.eval()
        self.mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
        self.num_vq_tokens = int(config.model.showo.num_vq_tokens)
        self.codebook_size = int(config.model.showo.codebook_size)
        self.num_new_special_tokens = int(config.model.showo.num_new_special_tokens)
        self.llm_vocab_size = int(config.model.showo.llm_vocab_size)
        self.token_offset = self.llm_vocab_size + self.num_new_special_tokens
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
        image = images[0].permute(1, 2, 0).detach().cpu().numpy()
        image = (image * 255.0).round().clip(0, 255).astype("uint8")
        self.Image.fromarray(image).save(output_path)
        return output_path

    def _tokens_to_prompt_ids(self, model_tokens):
        self.load()
        tokens = self.torch.tensor(model_tokens, dtype=self.torch.long, device=self.device).reshape(1, -1)
        return self.torch.where(tokens == self.mask_token_id, tokens, tokens + self.token_offset)

    def _attention(self, input_ids, uncond_input_ids):
        if uncond_input_ids is not None:
            ids = self.torch.cat([input_ids, uncond_input_ids], dim=0)
        else:
            ids = input_ids
        return self.create_attention_mask_predict_next(
            ids,
            pad_id=int(self.uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(self.uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(self.uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        )

    def run_confidence_block(self, prompt, model_tokens=None, steps=None, seed=None):
        self.load()
        if model_tokens is None:
            model_tokens = self.empty_model_tokens()
        steps = int(steps or self.generation_timesteps)
        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        image_prompt_tokens = self._tokens_to_prompt_ids(model_tokens)
        input_ids, _ = self.uni_prompting(([prompt], image_prompt_tokens), "t2i_gen")
        if self.guidance_scale > 0:
            uncond_input_ids, _ = self.uni_prompting(([""], image_prompt_tokens), "t2i_gen")
        else:
            uncond_input_ids = None
        attention_mask = self._attention(input_ids, uncond_input_ids)
        uncond_prefix = None
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :self.config.dataset.preprocessing.max_seq_length + 1]
        model_tokens_tensor = self.torch.tensor(model_tokens, dtype=self.torch.long, device=self.device).reshape(1, -1)
        last_sampled_ids = self.torch.where(model_tokens_tensor == self.mask_token_id, self.torch.zeros_like(model_tokens_tensor), model_tokens_tensor)
        last_confidence = self.torch.zeros((1, self.num_vq_tokens), dtype=self.torch.float32, device=self.device)
        last_confidence_mask = self.torch.zeros((1, self.num_vq_tokens), dtype=self.torch.bool, device=self.device)
        temperature = float(self.config.training.get("generation_temperature", 1.0))
        step_records = []
        with self.torch.no_grad():
            for step in range(steps):
                if uncond_input_ids is not None and self.guidance_scale > 0:
                    uncond_input_ids = self.torch.cat([uncond_prefix, input_ids[:, self.config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                    model_input = self.torch.cat([input_ids, uncond_input_ids], dim=0)
                    cond_logits, uncond_logits = self.model(model_input, attention_mask=attention_mask).chunk(2)
                    logits = (1 + self.guidance_scale) * cond_logits - self.guidance_scale * uncond_logits
                else:
                    logits = self.model(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(self.num_vq_tokens + 1):-1, self.token_offset:-1]
                probs = logits.softmax(dim=-1)
                sampled = probs.reshape(-1, logits.size(-1))
                sampled_ids = self.torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
                unknown_map = model_tokens_tensor == self.mask_token_id
                sampled_ids = self.torch.where(unknown_map, sampled_ids, model_tokens_tensor)
                selected_probs = self.torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)
                selected_probs = self.torch.where(unknown_map, selected_probs, self.torch.finfo(selected_probs.dtype).max)
                unknown_count = int(unknown_map.sum().item())
                ratio = float(step + 1) / float(max(1, steps))
                if unknown_count > 1:
                    mask_ratio = self.mask_schedule(self.torch.tensor(ratio, device=self.device))
                    mask_len = (self.num_vq_tokens * mask_ratio).floor().reshape(1, 1).to(self.device)
                    mask_len = self.torch.max(
                        self.torch.tensor([1], device=self.device),
                        self.torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len),
                    )
                    temperature = temperature * (1.0 - ratio)
                    confidence_mask = self.mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
                else:
                    confidence_mask = self.torch.zeros_like(unknown_map)
                model_tokens_tensor = self.torch.where(confidence_mask, self.mask_token_id, sampled_ids)
                input_ids[:, -(self.num_vq_tokens + 1):-1] = self.torch.where(
                    confidence_mask,
                    self.mask_token_id,
                    sampled_ids + self.token_offset,
                )
                last_sampled_ids = sampled_ids
                last_confidence = selected_probs.float()
                last_confidence_mask = confidence_mask
                step_records.append({
                    "step": step,
                    "unknown_before": unknown_count,
                    "confidence_remask_count": int(confidence_mask.sum().item()),
                    "mean_confidence": float(selected_probs[unknown_map].mean().item()) if unknown_count else 1.0,
                })
        return {
            "model_tokens": model_tokens_tensor.reshape(-1).detach().cpu().long().tolist(),
            "decoded_tokens": last_sampled_ids.reshape(-1).detach().cpu().long().tolist(),
            "confidence": last_confidence.reshape(-1).detach().cpu().float().tolist(),
            "confidence_mask": last_confidence_mask.reshape(-1).detach().cpu().bool().tolist(),
            "confidence_steps": steps,
            "step_records": step_records,
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

    def answer_image(self, question, image_path, max_new_tokens=192, top_k=1):
        self.load()
        image_tokens = self.torch.tensor(self.encode_image(image_path), dtype=self.torch.long, device=self.device).reshape(1, -1)
        image_tokens = image_tokens + self.token_offset
        question_ids = self.uni_prompting.text_tokenizer(["USER: \n" + question + " ASSISTANT:"])["input_ids"]
        question_ids = self.torch.tensor(question_ids, dtype=self.torch.long, device=self.device)
        input_ids = self.torch.cat([
            (self.torch.ones(question_ids.shape[0], 1, device=self.device) * self.uni_prompting.sptids_dict["<|mmu|>"].to(self.device)).long(),
            (self.torch.ones(question_ids.shape[0], 1, device=self.device) * self.uni_prompting.sptids_dict["<|soi|>"].to(self.device)).long(),
            image_tokens,
            (self.torch.ones(question_ids.shape[0], 1, device=self.device) * self.uni_prompting.sptids_dict["<|eoi|>"].to(self.device)).long(),
            (self.torch.ones(question_ids.shape[0], 1, device=self.device) * self.uni_prompting.sptids_dict["<|sot|>"].to(self.device)).long(),
            question_ids,
        ], dim=1)
        attention_mask = self.create_attention_mask_for_mmu(input_ids, eoi_id=int(self.uni_prompting.sptids_dict["<|eoi|>"]))
        with self.torch.no_grad():
            cont_toks = self.model.mmu_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                top_k=top_k,
                eot_token=self.uni_prompting.sptids_dict["<|eot|>"],
            )
        if not cont_toks:
            return ""
        cont_toks = self.torch.stack(cont_toks).reshape(1, -1)
        return self.uni_prompting.text_tokenizer.batch_decode(cont_toks, skip_special_tokens=True)[0]


def flat_to_grid(tokens, grid_size):
    return [list(tokens[row * grid_size:(row + 1) * grid_size]) for row in range(grid_size)]


def compact_token_payload(payload):
    return {
        "model_tokens": payload.get("model_tokens"),
        "decoded_tokens": payload.get("decoded_tokens"),
        "confidence_steps": payload.get("confidence_steps"),
        "step_records": payload.get("step_records"),
        "mask_token_id": payload.get("mask_token_id"),
    }
