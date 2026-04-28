from ascr.generators.base import MockGeneratorAdapter
from ascr.generators.showo import ShowOAdapter


def build_generator(name, config):
    name = (name or "mock").lower()
    config = config or {}
    if name == "mock":
        return MockGeneratorAdapter(token_grid_size=int(config.get("token_grid_size", 16)), image_size=int(config.get("image_size", 256)))
    if name == "showo":
        generator_config = config.get("generator", config)
        return ShowOAdapter(
            repo_path=generator_config.get("repo_path"),
            checkpoint_path=generator_config.get("checkpoint_path"),
            device=generator_config.get("device", "cuda"),
            token_grid_size=int(config.get("token_grid_size", 16)),
            image_size=int(config.get("image_size", 256)),
        )
    raise ValueError(f"Unknown generator: {name}")
