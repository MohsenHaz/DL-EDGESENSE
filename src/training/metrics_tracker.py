"""Track training metrics across epochs."""
class MetricsTracker:
    def __init__(self):
        self.history = []
    def update(self, epoch, metrics):
        self.history.append({"epoch": epoch, **metrics})
    def best(self, key="f1_weighted"):
        return max(self.history, key=lambda x: x[key])
