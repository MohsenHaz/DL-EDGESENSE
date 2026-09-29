"""Elastic Weight Consolidation (EWC) — Section 4.4.10."""
import torch, torch.nn as nn

class EWC:
    def __init__(self, model, dataloader, device="cpu", lambda_ewc=5000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader, device)

    def _compute_fisher(self, loader, device):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in loader:
            inputs, targets = batch["inputs"], batch["labels"].to(device)
            logits, _ = self.model(inputs)
            loss = nn.CrossEntropyLoss()(logits, targets)
            self.model.zero_grad(); loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(loader)
        return fisher

    def penalty(self, model) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return 0.5 * self.lambda_ewc * loss
