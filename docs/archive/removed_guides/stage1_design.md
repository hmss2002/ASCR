# Stage 1 Design Notes

Stage 1 validates ASCR with a zero-training interface. The grid and JSON exchange are engineering devices for the first prototype, while the method object is selective semantic reopening.

## Required Properties

- Keep the ASCR loop independent from Show-o-specific code.
- Treat malformed evaluator output as an abstention.
- Preserve traces that can later supervise a learned selector.
- Keep all paths configurable so long-running jobs can move between interactive and batch nodes.

## Main Interfaces

- GeneratorAdapter: wraps Show-o or mock generation.
- SemanticEvaluator: wraps local VLM or mock evaluation.
- SemanticReopeningSelector: converts semantic reports into token masks.
- TraceWriter: writes Stage 2 training examples.
