import argparse
import json
from datetime import datetime
from pathlib import Path

from ascr.benchmarks.metrics import compare_scores, score_image
from ascr.benchmarks.runner import result_to_markdown
from ascr.core.config import load_config
from ascr.core.loop import ASCRLoop, run_config_from_mapping
from ascr.core.state import GenerationState
from ascr.evaluators.registry import build_evaluator
from ascr.generators.registry import build_generator
from ascr.generators.showo import ShowOAdapter
from ascr.revision.selector import GridSemanticReopeningSelector


def build_parser():
    parser = argparse.ArgumentParser(description='Compare original Show-o against ASCR Stage 1.')
    parser.add_argument('--config', default='configs/stage1_showo_local.yaml')
    parser.add_argument('--prompt', default='A red cube left of a blue sphere')
    parser.add_argument('--output-dir', default='outputs/benchmarks')
    parser.add_argument('--generation-timesteps', type=int, default=None)
    parser.add_argument('--guidance-scale', type=float, default=None)
    parser.add_argument('--max-iterations', type=int, default=2)
    return parser


def build_loop(config):
    generator_config = dict(config)
    generator_config['token_grid_size'] = int(config.get('token_grid_size', 32))
    generator_config['image_size'] = int(config.get('image_size', 512))
    generator = build_generator(config.get('generator', {}).get('name', 'showo'), generator_config)
    evaluator = build_evaluator(config.get('evaluator', {}).get('name', 'local_vlm'), config)
    selector = GridSemanticReopeningSelector(
        coarse_grid_size=int(config.get('coarse_grid_size', 4)),
        token_grid_size=int(config.get('token_grid_size', 32)),
        dilation=int(config.get('dilation', 1)),
    )
    return ASCRLoop(generator, evaluator, selector, run_config_from_mapping(config))


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config['max_iterations'] = args.max_iterations
    generator_config = config.setdefault('generator', {})
    if args.generation_timesteps is not None:
        generator_config['generation_timesteps'] = args.generation_timesteps
    if args.guidance_scale is not None:
        generator_config['guidance_scale'] = args.guidance_scale
    root = Path(args.output_dir) / datetime.utcnow().strftime('showo_ascr-%Y%m%d-%H%M%S')
    root.mkdir(parents=True, exist_ok=True)
    baseline_path = root / 'baseline_showo.png'
    baseline_generator = ShowOAdapter(
        repo_path=generator_config.get('repo_path'),
        checkpoint_path=generator_config.get('checkpoint_path'),
        vq_model_path=generator_config.get('vq_model_path'),
        llm_model_path=generator_config.get('llm_model_path'),
        showo_config_path=generator_config.get('showo_config_path'),
        device=generator_config.get('device', 'cuda'),
        token_grid_size=int(config.get('token_grid_size', 32)),
        image_size=int(config.get('image_size', 512)),
        guidance_scale=float(generator_config.get('guidance_scale', 4.0)),
        generation_timesteps=int(generator_config.get('generation_timesteps', 18)),
        seed=int(config.get('seed', generator_config.get('seed', 1234))),
    )
    baseline_generator.generate_t2i(args.prompt, baseline_path)
    baseline_state = GenerationState(
        prompt=args.prompt,
        iteration=0,
        token_grid=[[0 for _ in range(int(config.get("token_grid_size", 32)))] for _ in range(int(config.get("token_grid_size", 32)))],
        image_path=str(baseline_path),
        metadata={"generator": "showo", "source": "baseline_compare"},
    )
    config['output_dir'] = str(root / 'ascr')
    config['run_name'] = 'stage1_showo_ascr'
    summary = build_loop(config).run(args.prompt, project_root=Path.cwd(), initial_state=baseline_state)
    grid_size = int(config.get('coarse_grid_size', 4))
    image_size = int(config.get('image_size', 512))
    baseline_score = score_image(args.prompt, baseline_path, grid_size=grid_size, image_size=image_size)
    ascr_score = score_image(args.prompt, summary['final_decoded_image'], grid_size=grid_size, image_size=image_size)
    comparison = compare_scores(baseline_score, ascr_score)
    result = {
        'prompt': args.prompt,
        'baseline_image': str(baseline_path),
        'ascr_final_image': summary['final_decoded_image'],
        'ascr_summary': summary,
        'baseline_score': baseline_score,
        'ascr_score': ascr_score,
        'comparison': comparison,
    }
    result_path = root / 'comparison.json'
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + chr(10), encoding='utf-8')
    markdown_path = root / 'comparison.md'
    markdown_path.write_text(result_to_markdown(result), encoding='utf-8')
    print(json.dumps({'result_path': str(result_path), 'markdown_path': str(markdown_path), 'comparison': comparison}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
