# Stage 2 interface placeholder.
# This class is intentionally left unimplemented. The Stage 1 loop and trace writer
# produce the (prompt, grid localization, token mask, correction outcome) training
# examples that Stage 2 will need. When Stage 2 begins, this class should be
# implemented as a lightweight decision head that maps (image features, prompt
# embedding) → token-level reopening scores, replacing GridSemanticReopeningSelector.
class SemanticReopeningSelectorModel:
    def __init__(self):
        raise NotImplementedError("Stage 2 learned selector model is reserved for the next project stage.")
