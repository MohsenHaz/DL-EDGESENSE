"""Early stopping."""
class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best = None

    def __call__(self, score):
        if self.best is None or score > self.best:
            self.best = score; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience
