"""Metric-aware decoding: turn calibrated pair probabilities into per-S1 match lists.

1. Partition constraint: an S2/S3 record can belong to at most one S1 entity, so each record
   is kept only for the S1 that gives it the highest probability.
2. Expected-F0.5 decoding: for every S1 we sort its candidates by probability and pick the
   prefix length k (k = 0 means "singleton") that maximises the Monte-Carlo estimate of
   E[F0.5]. Matches that blocking missed (a Poisson term, lam_miss) are included in the
   estimate. The macro-averaged per-entity metric, singletons included, is exactly what
   this optimises, rather than a single global threshold.
"""
import numpy as np
import pandas as pd

BETA2 = 0.25


def enforce_partition(pred: pd.DataFrame, col="p") -> pd.DataFrame:
    best = pred.groupby("m")[col].transform("max")
    keep = pred[col] >= best
    out = pred[keep]
    return out.drop_duplicates("m", keep="first")


def expected_f_decode(pred: pd.DataFrame, col="p", max_list=12, n_samples=256, lam_miss=0.0,
                      p_floor=0.02, seed=0, chunk=20_000) -> pd.DataFrame:
    """pred: columns s1, m, p. Returns the rows selected as matches."""
    rng = np.random.default_rng(seed)
    d = pred[pred[col] >= p_floor].sort_values(["s1", col], ascending=[True, False])
    d = d[d.groupby("s1").cumcount() < max_list]
    if d.empty:
        return d
    s1_codes, s1_uni = pd.factorize(d.s1, sort=False)
    pos = d.groupby("s1").cumcount().to_numpy()
    G, L = len(s1_uni), max_list
    P = np.zeros((G, L), dtype=np.float32)
    P[s1_codes, pos] = d[col].to_numpy(np.float32)
    best_k = np.zeros(G, dtype=np.int16)
    ks = np.arange(L + 1, dtype=np.float32)
    for i in range(0, G, chunk):
        p = P[i:i + chunk]                                      # (g, L)
        X = rng.random((p.shape[0], n_samples, L), dtype=np.float32) < p[:, None, :]
        tp = np.concatenate([np.zeros(X.shape[:2] + (1,), np.float32), np.cumsum(X, axis=2, dtype=np.float32)], axis=2)
        T = tp[..., -1:]
        if lam_miss > 0:
            T = T + rng.poisson(lam_miss, size=T.shape).astype(np.float32)
        denom = BETA2 * T + ks[None, None, :]
        F = np.where(denom > 0, (1 + BETA2) * tp / np.maximum(denom, 1e-9), 0.0)
        F[..., 0] = (T[..., 0] == 0)                             # empty prediction scores 1 iff no true match
        best_k[i:i + chunk] = F.mean(axis=1).argmax(axis=1)
    sel = pos < best_k[s1_codes]
    return d[sel]


def threshold_decode(pred: pd.DataFrame, thr: float, col="p") -> pd.DataFrame:
    return pred[pred[col] >= thr]


def macro_f05(selected: pd.DataFrame, gt_pairs: pd.DataFrame, s1_ids: np.ndarray) -> float:
    """Exact challenge metric on a set of S1 ids. selected/gt_pairs: columns s1, m."""
    s1_ids = np.asarray(s1_ids)
    gt = gt_pairs[gt_pairs.s1.isin(s1_ids)][["s1", "m"]]
    sel = selected[selected.s1.isin(s1_ids)][["s1", "m"]]
    tp = sel.merge(gt, on=["s1", "m"]).groupby("s1").size()
    n_pred = sel.groupby("s1").size()
    n_true = gt.groupby("s1").size()
    df = pd.DataFrame(index=pd.Index(s1_ids, name="s1"))
    df["tp"] = tp.reindex(df.index).fillna(0).to_numpy()
    df["np"] = n_pred.reindex(df.index).fillna(0).to_numpy()
    df["nt"] = n_true.reindex(df.index).fillna(0).to_numpy()
    f = np.where((df.np == 0) & (df.nt == 0), 1.0,
                 np.where(df.tp == 0, 0.0, (1 + BETA2) * df.tp / (BETA2 * df.nt + df.np).clip(lower=1e-9)))
    return float(f.mean())
