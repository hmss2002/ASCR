from ascr.generators.base import GeneratorAdapter


class ShowOAdapter(GeneratorAdapter):
    def __init__(self, repo_path=None, checkpoint_path=None, device="cuda", token_grid_size=16, image_size=256):
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.token_grid_size = token_grid_size
        self.image_size = image_size

    def _not_connected(self):
        raise NotImplementedError(
            "Show-o integration is reserved behind GeneratorAdapter. Set generator=mock for dry runs or provide Show-o repo and checkpoint paths before real Stage 1 inference."
        )

    def initialize(self, prompt, artifacts):
        self._not_connected()

    def decode(self, state, output_path):
        self._not_connected()

    def reopen_and_continue(self, state, mask, correction_prompt, artifacts):
        self._not_connected()
