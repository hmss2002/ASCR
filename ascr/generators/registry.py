from ascr.generators.base import MockGeneratorAdapter


def _generator_config(config):
    return (config or {}).get('generator', config or {})


def build_generator(name, config):
    name = (name or 'mock').lower()
    config = config or {}
    if name == 'mock':
        return MockGeneratorAdapter(token_grid_size=int(config.get('token_grid_size', 16)), image_size=int(config.get('image_size', 256)))
    if name == 'lumina':
        from ascr.generators.lumina_dimoo import LuminaAdapter
        generator_config = _generator_config(config)
        return LuminaAdapter(
            repo_path=generator_config.get('repo_path'),
            checkpoint_path=generator_config.get('checkpoint_path'),
            lora_path=generator_config.get('lora_path'),
            device=generator_config.get('device', 'cuda'),
            token_grid_size=int(config.get('token_grid_size', generator_config.get('token_grid_size', 64))),
            image_size=int(config.get('image_size', generator_config.get('image_size', 1024))),
            guidance_scale=float(generator_config.get('guidance_scale', 4.0)),
            generation_timesteps=int(generator_config.get('generation_timesteps', 64)),
            temperature=float(generator_config.get('temperature', 1.0)),
            seed=int(config.get('seed', generator_config.get('seed', 1234))),
            answer_steps=int(generator_config.get('answer_steps', 64)),
            answer_block_length=int(generator_config.get('answer_block_length', 128)),
            answer_temperature=float(generator_config.get('answer_temperature', 0.0)),
            answer_cfg_scale=float(generator_config.get('answer_cfg_scale', 0.0)),
        )
    if name == 'mmada':
        from ascr.generators.mmada import MMaDAAdapter
        generator_config = _generator_config(config)
        return MMaDAAdapter(
            repo_path=generator_config.get('repo_path'),
            checkpoint_path=generator_config.get('checkpoint_path'),
            vq_model_path=generator_config.get('vq_model_path'),
            device=generator_config.get('device', 'cuda'),
            token_grid_size=int(config.get('token_grid_size', generator_config.get('token_grid_size', 32))),
            image_size=int(config.get('image_size', generator_config.get('image_size', 512))),
            guidance_scale=float(generator_config.get('guidance_scale', 3.5)),
            generation_timesteps=int(generator_config.get('generation_timesteps', 15)),
            seed=int(config.get('seed', generator_config.get('seed', 1234))),
            max_seq_length=int(generator_config.get('max_seq_length', 512)),
        )
    if name == 'showo':
        from ascr.generators.showo import ShowOAdapter

        generator_config = _generator_config(config)
        return ShowOAdapter(
            repo_path=generator_config.get('repo_path'),
            checkpoint_path=generator_config.get('checkpoint_path'),
            vq_model_path=generator_config.get('vq_model_path'),
            llm_model_path=generator_config.get('llm_model_path'),
            showo_config_path=generator_config.get('showo_config_path'),
            device=generator_config.get('device', 'cuda'),
            token_grid_size=int(config.get('token_grid_size', generator_config.get('token_grid_size', 32))),
            image_size=int(config.get('image_size', generator_config.get('image_size', 512))),
            guidance_scale=float(generator_config.get('guidance_scale', 4.0)),
            generation_timesteps=int(generator_config.get('generation_timesteps', 50)),
            seed=int(config.get('seed', generator_config.get('seed', 1234))),
            native_token_loop=bool(generator_config.get('native_token_loop', True)),
            confidence_steps=int(generator_config.get('confidence_steps', 50)),
        )
    raise ValueError(f'Unknown generator: {name}')
