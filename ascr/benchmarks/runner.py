def run_smoke_benchmark(loop_factory, prompts):
    results = []
    for subset, subset_prompts in prompts.items():
        for prompt in subset_prompts:
            summary = loop_factory().run(prompt)
            results.append({"subset": subset, "prompt": prompt, "summary": summary})
    return results
