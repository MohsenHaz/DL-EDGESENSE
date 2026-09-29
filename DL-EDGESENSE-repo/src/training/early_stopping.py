"""Early stopping on the validation energy-accuracy ratio, patience=10
per Table S2 / configs/training.yaml."""


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = metric
            return False

        improved = (metric > self.best + self.min_delta if self.mode == "max"
                    else metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience
