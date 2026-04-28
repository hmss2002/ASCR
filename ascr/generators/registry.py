from ascr.generators.base import MockGeneratorAdapter
from ascr.generators.showo import ShowOAdapter


def build_generator(name, config):
    name = (name or 'mock').lower()
    config = config or {}
    if name == 'mock':
        return MockGeneratorAdapter(token_grid_size=int(config.get('token_grid_size', 16)), image_size=int(config.get('image_size', 256)))
    if name == 'showo':
        generator_config = config.get('generator', config)
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
            generation_timesteps=int(generator_config.get('generation_timesteps', 18)),
            seed=int(config.get('seed', generator_config.get('seed', 1234))),
            native_token_loop=bool(generator_config.get('native_token_loop', True)),
            confidence_steps=int(generator_config.get('confidence_steps', 2)),
        )
    raise ValueError(f'Unknown generator: {name}')
