def result_to_markdown(result):
    comparison = result['comparison']
    lines = [
        '| Prompt | Baseline | ASCR | Delta | Verdict |',
        '| --- | ---: | ---: | ---: | --- |',
        f'| {result["prompt"]} | {comparison["baseline_score"]:.3f} | {comparison["ascr_score"]:.3f} | {comparison["delta"]:.3f} | {comparison["verdict"]} |',
    ]
    return chr(10).join(lines) + chr(10)
