def compose_correction_prompt(original_prompt, evaluation):
    parts = [original_prompt.strip()]
    if evaluation.correction_instruction:
        parts.append(f"ASCR correction: {evaluation.correction_instruction.strip()}")
    elif evaluation.summary:
        parts.append(f"ASCR correction: repair the localized semantic issue: {evaluation.summary.strip()}")
    else:
        parts.append("ASCR correction: repair the localized semantic issue while preserving correct regions.")
    return chr(10).join(parts)
