"""Run-level statistics. Never pass correlated ticks as independent replicates."""
import itertools
import math
import random
from statistics import fmean, stdev, median

ALPHA = .05
EXACT_LIMIT = 16
MC_DRAWS = 100_000
MC_SEED = 350037


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def quantile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("Quantile requires data and q in [0,1]")
    values = sorted(values)
    p = q * (len(values) - 1)
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] + (p - lo) * (values[hi] - values[lo])


def mean_interval(values, confidence=.95):
    """Student-t interval for a mean of independent, approximately normal units."""
    values = list(values)
    if not 0 < confidence < 1 or any(not finite(v) for v in values):
        raise ValueError("Invalid confidence level or non-finite observation")
    if len(values) < 2:
        return None
    from scipy.stats import t
    center = fmean(values)
    half = float(t.ppf((1 + confidence) / 2, len(values) - 1)) * stdev(values) / math.sqrt(len(values))
    return [center - half, center + half]


def describe(values):
    """Explicit missingness. No imputation; intervals use available units only."""
    raw = list(values)
    if any(v is not None and not finite(v) for v in raw):
        raise ValueError("Use finite numeric values or explicit None")
    valid = [v for v in raw if v is not None]
    sd = stdev(valid) if len(valid) >= 2 else None
    return {"n": len(valid), "n_expected": len(raw), "n_missing": len(raw)-len(valid),
            "values": raw, "mean": fmean(valid) if valid else None,
            "sample_sd": sd, "median": median(valid) if valid else None,
            "q25": quantile(valid, .25) if valid else None,
            "q75": quantile(valid, .75) if valid else None,
            "min": min(valid) if valid else None, "max": max(valid) if valid else None,
            "mean_ci95": mean_interval(valid),
            "zero_sample_variance": sd == 0 if sd is not None else None}


def sign_flip(differences):
    """Two-sided mean-difference sign-flip test.

    Exact under independent, sign-symmetric differences (or genuine
    within-pair exchangeability under a randomization null). A shared
    seed label alone does not establish these assumptions.
    """
    values = list(differences)
    if not values or any(not finite(v) for v in values):
        raise ValueError("Require nonempty finite paired differences")
    nonzero = [v for v in values if v != 0]
    observed = abs(math.fsum(nonzero))
    tol = 1e-12 * math.fsum(abs(v) for v in nonzero)
    if len(nonzero) <= EXACT_LIMIT:
        permutations = 2**len(nonzero)
        hits = sum(abs(math.fsum(s*v for s,v in zip(signs, nonzero))) >= observed-tol
                   for signs in itertools.product((-1,1), repeat=len(nonzero)))
        return {"p_value": hits/permutations, "method": "exact_sign_flip",
                "permutations": permutations, "nonzero_pairs": len(nonzero)}
    rng = random.Random(MC_SEED)
    hits = sum(abs(math.fsum((v if rng.getrandbits(1) else -v) for v in nonzero)) >= observed-tol
               for _ in range(MC_DRAWS))
    return {"p_value": (hits+1)/(MC_DRAWS+1), "method": "monte_carlo_sign_flip",
            "permutations": MC_DRAWS, "rng_seed": MC_SEED, "nonzero_pairs": len(nonzero)}


def paired(candidate, reference, relative=False, test=False):
    """Aligned per-run values, candidate minus reference, no pairwise deletion for inference."""
    candidate, reference = list(candidate), list(reference)
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("Require equally sized nonempty matched arms")
    a, b = describe(candidate), describe(reference)
    deltas = [x-y if x is not None and y is not None else None for x,y in zip(candidate,reference)]
    diff = describe(deltas)
    complete = diff["n_missing"] == 0
    reduction = None
    if relative:
        if any(v is not None and v < 0 for v in candidate+reference):
            raise ValueError("Relative reduction only for nonnegative physical metrics")
        if complete and b["mean"] > 0:
            reduction = 100*(b["mean"]-a["mean"])/b["mean"]
    return {"candidate": a, "reference": b, "candidate_minus_reference": diff,
            "complete_pairs": complete, "n_pairs": diff["n"],
            "relative_reduction_pct": reduction,
            "relative_definition": "100*(mean(reference)-mean(candidate))/mean(reference)" if relative else None,
            "test": sign_flip(deltas) if test and complete and len(deltas) >= 2 else None,
            "inference_status": ("complete" if complete and len(deltas) >= 2
                                 else "insufficient_or_missing_pairs")}


def holm(p_values, alpha=ALPHA):
    """Holm step-down family-wise correction, retaining untestable hypotheses.

    None is conservatively assigned p=1 in the ranking but never
    reported as a tested/rejected hypothesis. Caller may suppress the
    entire confirmatory family when missing endpoints remain.
    """
    values = list(p_values)
    if not values or not 0 < alpha < 1 or any(p is not None and (not finite(p) or not 0 <= p <= 1) for p in values):
        raise ValueError("Invalid p-value family")
    ordered = sorted(range(len(values)), key=lambda i: (1 if values[i] is None else values[i], i))
    adjusted = [None]*len(values)
    previous = 0.0
    for rank,i in enumerate(ordered):
        p = 1 if values[i] is None else values[i]
        previous = min(1.0, max(previous, (len(values)-rank)*p))
        if values[i] is not None:
            adjusted[i] = previous
    return [{"p_raw": p, "p_holm": q, "reject": q is not None and q <= alpha}
            for p,q in zip(values,adjusted)]


def equivalence(differences, margin):
    """Exploratory paired TOST via a 90% t interval; margin must be prespecified."""
    if not finite(margin) or margin <= 0:
        raise ValueError("Equivalence margin must be positive and finite")
    summary = describe(differences)
    valid = summary["n"] >= 2 and summary["n_missing"] == 0
    interval = mean_interval(summary["values"], .90) if valid else None
    # Constant differences do not establish the variance model from data.
    return {"margin": margin, "difference": summary, "mean_ci90": interval,
            "within_margin": bool(interval is not None and not summary["zero_sample_variance"]
                                  and -margin < interval[0] and interval[1] < margin),
            "status": "exploratory_tost_requires_independent_approximately_normal_differences"}
