"""Paired t-tests + Bonferroni correction — as used in paper."""
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

def paired_t_bonferroni(scores_a: list, scores_b: list, alpha=0.05):
    """Run paired t-test and apply Bonferroni correction."""
    t_stat, p_val = ttest_rel(scores_a, scores_b)
    reject, p_corr, _, _ = multipletests([p_val], alpha=alpha, method="bonferroni")
    ci = (np.mean(np.array(scores_a) - np.array(scores_b))
          - 1.96 * np.std(np.array(scores_a) - np.array(scores_b)) / np.sqrt(len(scores_a)),
          np.mean(np.array(scores_a) - np.array(scores_b))
          + 1.96 * np.std(np.array(scores_a) - np.array(scores_b)) / np.sqrt(len(scores_a)))
    return {"t": t_stat, "p": p_val, "p_bonferroni": p_corr[0],
            "reject_h0": bool(reject[0]), "ci_95": ci}
