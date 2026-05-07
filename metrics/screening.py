"""
metrics/screening.py
TopU-LBVS Setting 1 - Screening evaluation metrics.
All metrics take:
    y_true  : array-like of int   (1=active, 0=inactive/decoy)
    scores  : array-like of float (higher score = more likely active)

Metric definitions follow Appendix D of the TopU-LBVS paper exactly.

Functions
---------
ef(y_true, scores, percent)         EF@percent%
prauc(y_true, scores)               PR-AUC (average precision)
rocauc(y_true, scores)              ROC-AUC
bedroc(y_true, scores, alpha=20.0)  BEDROC
logauc(y_true, scores, min_fpr)     LogAUC (log-scale ROC AUC)
compute_all(y_true, scores)         dict with all metrics at once

Edge cases (handled silently, no crash):
    - y_true all zeros (no actives)         returns 0.0 for all metrics
    - y_true all ones  (no decoys)          returns 1.0 for BEDROC, 0.0 others
    - fewer compounds than k in EF          k clamped to N
    - single unique score value             returns 0.0 for AUC metrics
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# EF@x%  (Enrichment Factor)
# ---------------------------------------------------------------------------

def ef(y_true, scores, percent: float) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    N = len(y_true)
    Npos = float(y_true.sum())
    if Npos == 0 or N == 0:
        return 0.0
    k = max(1, min(int(np.ceil(percent / 100.0 * N)), N))

    # Sort descending; equal scores form contiguous tie groups
    order = np.argsort(-scores, kind='stable')
    y_sorted = y_true[order]
    scores_sorted = scores[order]

    # Walk through tie groups
    H = 0.0
    i = 0
    while i < N:
        j = i + 1
        while j < N and scores_sorted[j] == scores_sorted[i]:
            j += 1
        # Group occupies 1-indexed ranks [i+1, j], size t = j - i
        r_lo, r_hi, t = i + 1, j, j - i
        if r_hi <= k:
            prob = 1.0
        elif r_lo > k:
            prob = 0.0
        else:
            prob = (k - r_lo + 1) / t
        H += prob * float(y_sorted[i:j].sum())
        i = j

    return float(H * N) / float(k * Npos)


# ---------------------------------------------------------------------------
# PR-AUC  (Precision-Recall Area Under Curve)
# ---------------------------------------------------------------------------

def prauc(y_true, scores) -> float:
    """
    Precision-Recall AUC (Average Precision).

    Uses sklearn's average_precision_score which computes the area under
    the precision-recall curve using the step interpolation (trapezoidal
    method on the recall-precision pairs from all thresholds).

    More informative than ROC-AUC under strong class imbalance (1:40 ratio).
    Used as the validation metric for model selection in Setting 1.

    Returns
    -------
    float in [0, 1]. Returns 0.0 if no actives or no decoys in y_true.-
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    Npos = int(y_true.sum())
    Nneg = int((1 - y_true).sum())

    if Npos == 0 or Nneg == 0:
        return 0.0

    # sklearn raises ValueError if only one class present - already guarded above
    return float(average_precision_score(y_true, scores))


# ---------------------------------------------------------------------------
# ROC-AUC  (Receiver Operating Characteristic AUC)
# ---------------------------------------------------------------------------

def rocauc(y_true, scores) -> float:
    """
    ROC Area Under Curve.

    Probability that a randomly chosen active is scored higher than a
    randomly chosen decoy. Reported in appendix only (legacy metric).

    Returns
    -------
    float in [0, 1]. Returns 0.0 if no actives or no decoys in y_true.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    Npos = int(y_true.sum())
    Nneg = int((1 - y_true).sum())

    if Npos == 0 or Nneg == 0:
        return 0.0

    return float(roc_auc_score(y_true, scores))


# ---------------------------------------------------------------------------
# BEDROC  (Boltzmann Enhanced Discrimination of ROC)
# ---------------------------------------------------------------------------

def bedroc(y_true, scores, alpha: float = 20.0) -> float:
    """
    BEDROC (Truchon & Bayly, 2007), evaluated at alpha=20 as per the paper.

    Formulation (paper Appendix D.4):
        rj   = rank of jth active (1 = top of list)
        R    = (1/N+) * sum_j exp(-alpha * rj / N)
        R_rand = expected R under random ranking (geometric series)
               = exp(-a/N) * (1 - exp(-alpha)) / (N * (1 - exp(-alpha/N)))
                 where a = exp(-alpha/N)
        R_max  = R when actives occupy ranks 1..N+ (best case)
               = exp(-a/N) * (1 - exp(-alpha*N+/N)) / (N+ * (1 - exp(-alpha/N)))
        BEDROC = (R - R_rand) / (R_max - R_rand)

    Scale: 0 = random ranking, 1 = perfect ranking.

    Parameters
    ----------
    alpha : float - exponential decay parameter, paper fixes alpha=20

    Returns
    -------
    float. Returns 0.0 if no actives, 1.0 if all compounds are active.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    N    = len(y_true)
    Npos = int(y_true.sum())

    if Npos == 0:
        return 0.0
    if Npos == N:
        return 1.0

    # 1-indexed ranks; ties get average rank (input-order-independent)
    ranks_all = rankdata(-scores, method='average')
    ranks = ranks_all[y_true == 1]   # shape (Npos,)

    # R: observed weighted sum
    R = np.sum(np.exp(-alpha * ranks / N)) / Npos

    # base value: a = exp(-alpha/N)
    a = np.exp(-alpha / N)

    # R_rand: expected R under random ranking
    # = (1/N) * sum_{r=1}^{N} exp(-alpha*r/N)
    # = a * (1 - a^N) / (N * (1 - a))
    R_rand = a * (1.0 - a ** N) / (N * (1.0 - a))

    # R_max: actives at ranks 1..Npos
    # = (1/Npos) * sum_{r=1}^{Npos} exp(-alpha*r/N)
    # = a * (1 - a^Npos) / (Npos * (1 - a))
    R_max = a * (1.0 - a ** Npos) / (Npos * (1.0 - a))

    denom = R_max - R_rand
    if abs(denom) < 1e-12:
        # degenerate: R_max  R_rand (only possible when alpha0 or N very small)
        return 0.0

    return float((R - R_rand) / denom)

def bedroc_rdkit(y_true, scores, alpha: float = 20.0) -> float:
    """
    Standard BEDROC via rdkit.ML.Scoring.CalcBEDROC.

    Uses the original Truchon & Bayly (2007) formulation:
        BEDROC = (R - R_min) / (R_max - R_min)

    Bounded in [0, 1] by construction. Random ranking gives a small
    positive value (~0.1 for N=1000, Npos=100, alpha=20). For cross-
    paper comparability with other LBVS benchmarks.

    Returns
    -------
    float in [0, 1]. Returns 0.0 if no actives, 1.0 if all are active.
    """
    from rdkit.ML.Scoring import Scoring
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return 0.0
    if y.sum() == len(y):
        return 1.0
    order = np.argsort(-s, kind='stable')
    pairs = [[float(s[i]), int(y[i])] for i in order]
    return float(Scoring.CalcBEDROC(pairs, col=1, alpha=alpha))

# ---------------------------------------------------------------------------
# LogAUC  (log-scale ROC AUC)
# ---------------------------------------------------------------------------

def logauc(y_true, scores, min_fpr: float = 0.001) -> float:
    """
    LogAUC - area under the ROC curve plotted on a log10 FPR axis,
    integrated from min_fpr to 1.0, normalized by log10(1/min_fpr).

    Standard LBVS LogAUC as used in the LBVS literature.
    Emphasizes early enrichment similar to EF and BEDROC.

    Implementation details:
        - Uses drop_intermediate=False to retain all threshold points
          (important for small datasets where sklearn prunes the curve)
        - Ensures curve reaches (1.0, 1.0) by appending endpoint if needed
        - Interpolates tpr at min_fpr boundary using linear interpolation
          between (0,0) and the first curve point
        - Integrates tpr over log10(fpr) using the trapezoidal rule
        - Normalizes by log10(1/min_fpr) = log10(1000) = 3.0

    Parameters
    ----------
    min_fpr : float - lower FPR cutoff for integration, default 0.001

    Returns
    -------
    float in [0, ~1]. Returns 0.0 if no actives or no decoys.
    Note: maximum achievable LogAUC < 1.0 for small libraries (finite
    FPR resolution). For large libraries (N >> 1000) it approaches 1.0
    under perfect ranking.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    Npos = int(y_true.sum())
    Nneg = int((1 - y_true).sum())

    if Npos == 0 or Nneg == 0:
        return 0.0

    # Get full ROC curve (drop_intermediate=False preserves all threshold points)
    fpr, tpr, _ = roc_curve(y_true, scores, drop_intermediate=False)

    # Ensure curve ends at (1.0, 1.0)
    if fpr[-1] < 1.0:
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)

    # Interpolate tpr at min_fpr if the curve starts above min_fpr
    # linear interpolation from (0, 0) to first curve point
    if fpr[0] > min_fpr:
        tpr_at_min = float(np.interp(min_fpr, [0.0, fpr[0]], [0.0, tpr[0]]))
        fpr = np.concatenate([[min_fpr], fpr])
        tpr = np.concatenate([[tpr_at_min], tpr])

    # Clip to [min_fpr, 1.0]
    mask  = fpr >= min_fpr
    fpr_c = fpr[mask]
    tpr_c = tpr[mask]

    if len(fpr_c) < 2:
        return 0.0

    # Integrate tpr over log10(fpr) using trapezoidal rule
    log_fpr = np.log10(fpr_c)
    # np.trapz for numpy <2.0, np.trapezoid for numpy >=2.0
    _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    auc_log = float(_trapz(tpr_c, log_fpr))

    # Normalize: divide by total log range = log10(1.0/min_fpr)
    norm = np.log10(1.0 / min_fpr)   # = 3.0 when min_fpr=0.001

    return auc_log / norm


# ---------------------------------------------------------------------------
# compute_all - runs all metrics at once for a single target evaluation
# ---------------------------------------------------------------------------

def compute_all(y_true, scores) -> dict:
    """
    Compute all 7 screening metrics for a single target * seed evaluation.

    Parameters
    ----------
    y_true : array-like of int   - ground truth labels (1=active, 0=decoy)
    scores : array-like of float - model predicted scores (higher = more active)

    Returns
    -------
    dict with keys:
        ef_1pct   : EF@1%   - primary metric for Setting 1
        ef_5pct   : EF@5%   - secondary
        ef_10pct  : EF@10%  - secondary
        prauc     : PR-AUC  - model selection + secondary
        rocauc    : ROC-AUC - legacy, appendix only
        bedroc    : BEDROC (alpha=20) - appendix only
        logauc    : LogAUC (min_fpr=0.001) - appendix only
        n_actives : int - number of actives in test set (for sanity checks)
        n_decoys  : int - number of decoys in test set
        n_total   : int - total compounds in test set
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    Npos = int(y_true.sum())
    Nneg = int((1 - y_true).sum())
    N    = Npos + Nneg
    min_fpr_used = max(0.001, 5.0 / max(Nneg, 1))
    
    return {
        # Primary metric (Setting 1)
        "ef_1pct":   ef(y_true, scores, 1.0),
        # Secondary metrics
        "ef_5pct":   ef(y_true, scores, 5.0),
        "ef_10pct":  ef(y_true, scores, 10.0),
        "prauc":     prauc(y_true, scores),
        # Legacy / appendix
        "rocauc":    rocauc(y_true, scores),
        "bedroc":    bedroc(y_true, scores, alpha=20.0),
         "bedroc_rdkit": bedroc_rdkit(y_true, scores, alpha=20.0), # standard
        "logauc":     logauc(y_true, scores, min_fpr=min_fpr_used),
        "logauc_min_fpr": min_fpr_used,   # log it

        # Counts for sanity checks and logging
        "n_actives": Npos,
        "n_decoys":  Nneg,
        "n_total":   N,
    }