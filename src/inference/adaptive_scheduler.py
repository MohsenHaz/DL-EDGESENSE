"""Adaptive scheduling — choose fusion mode based on energy budget."""

THRESHOLDS = {"eth1": 0.3, "eth2": 0.7}  # fraction of max budget

def choose_fusion(energy_available: float) -> str:
    """Equation 13 — energy-budget-driven fusion selection."""
    if energy_available < THRESHOLDS["eth1"]:
        return "early"
    elif energy_available < THRESHOLDS["eth2"]:
        return "hybrid"
    return "late"
