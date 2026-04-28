def run_smoke_benchmark(loop_factory, prompts):
    results = []
    for subset, subset_prompts in prompts.items():
        for prompt in subset_prompts:
            summary = loop_factory().run(prompt)
            results.append({'subset': subset, 'prompt': prompt, 'summary': summary})
    return results


def result_to_markdown(result):
    comparison = result['comparison']
    lines = [
        '| Prompt | Baseline | ASCR | Delta | Verdict |',
        '| --- | ---: | ---: | ---: | --- |',
        f'| {result["prompt"]} | {comparison["baseline_score"]:.3f} | {comparison["ascr_score"]:.3f} | {comparison["delta"]:.3f} | {comparison["verdict"]} |',
    ]
    return chr(10).join(lines) + chr(10)
