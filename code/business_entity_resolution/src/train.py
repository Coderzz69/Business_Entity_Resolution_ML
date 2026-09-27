"""Stage 4: train the two-stage matcher (XGBoost; CUDA when an NVIDIA GPU is available) on train candidates and evaluate on held-out S1 entities.

Split: S1 entities are hashed into TRAIN (fit), DEV (report). Stage A (pairwise GBDT) produces
out-of-fold probabilities on TRAIN via GroupKFold by S1, stage B adds list/coherence features
computed from those probabilities. Isotonic calibration and the decoder's lam_miss are fitted on
stage-B OOF predictions, and DEV is only used for the final report.
"""
import argparse, gc, pickle, time
from pathlib import Path

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from block import load_candidates
from decode import enforce_partition, expected_f_decode, macro_f05, threshold_decode
from features import RecordStore, list_features, pair_features
from io_utils import load_gt

XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", max_depth=8, eta=0.08, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, max_bin=256, nthread=12)
N_ROUNDS = 400


def pick_device() -> str:
    """'cuda' if XGBoost can train on an NVIDIA GPU, else 'cpu'."""
    try:
        xgb.train({"device": "cuda", "tree_method": "hist"}, xgb.DMatrix(np.array([[0.0], [1.0]]), label=[0, 1]), 1)
        return "cuda"
    except Exception:
        return "cpu"


def s1_bucket(ids, mod=100):
    return (np.asarray(ids) * 2654435761 % 2**32) % mod


def compute_features(cand, s1_store, pool_store, chunk=2_000_000):
    parts = []
    for i in range(0, len(cand), chunk):
        parts.append(pair_features(cand.iloc[i:i + chunk], s1_store, pool_store))
    return pd.concat(parts)


def fit_oof(X, y, groups, names, device, n_folds=3, rounds=N_ROUNDS):
    """X: float32 ndarray. Returns out-of-fold predictions and a booster fit on all rows (XGBoost, GPU if available)."""
    params = dict(XGB_PARAMS, device=device)
    oof = np.zeros(len(y), dtype=np.float32)
    for k, (tr, va) in enumerate(GroupKFold(n_folds).split(X, y, groups)):
        d = xgb.QuantileDMatrix(X[tr], y[tr], feature_names=names, max_bin=params["max_bin"])
        m = xgb.train(params, d, rounds)
        del d; gc.collect()
        m.set_param({"device": "cpu"})            # inputs are host arrays; avoid a GPU copy on predict
        oof[va] = m.inplace_predict(X[va])
        print(f"      fold {k}: done", flush=True)
    d = xgb.QuantileDMatrix(X, y, feature_names=names, max_bin=params["max_bin"])
    full = xgb.train(params, d, rounds)
    del d; gc.collect()
    full.set_param({"device": "cpu"})
    return oof, full


def main(data_dir: Path, work: Path, train_pct: int, dev_pct: int):
    t0 = time.time()
    pairs, _ = load_gt(data_dir)
    all_s1 = np.load(work / "s1ids_train.npy")          # S1 entities that went through blocking
    cand = load_candidates(work, "train", s1_filter=lambda s: s1_bucket(s) < train_pct + dev_pct)
    cand["is_train"] = s1_bucket(cand.s1.to_numpy()) < train_pct
    cand = cand.merge(pairs.assign(y=True), on=["s1", "m"], how="left")
    cand["y"] = cand.y.fillna(False).astype(bool)
    print(f"[{time.time() - t0:.0f}s] {len(cand):,} pairs, positives {cand.y.mean():.3f}", flush=True)

    s1 = pd.read_parquet(work / "n2_train_s1.parquet")
    s1_store = RecordStore([s1[s1.id.isin(cand.s1.unique())]])
    need = cand.m.unique()
    pool_store = RecordStore([pd.read_parquet(work / f"n2_train_s{k}.parquet").pipe(lambda d: d[d.id.isin(need)])
                              for k in (2, 3)])
    del s1
    X = compute_features(cand, s1_store, pool_store)
    print(f"[{time.time() - t0:.0f}s] pair features {X.shape}", flush=True)

    tr = cand.is_train.to_numpy()
    y = cand.y.to_numpy().astype(np.int8)
    groups = cand.s1.to_numpy()
    device = pick_device()
    print(f"   training device: {device}", flush=True)
    featA = list(X.columns)
    XA = X.to_numpy(dtype=np.float32)
    del X; gc.collect()
    # ---- stage A
    oofA, modelA = fit_oof(XA[tr], y[tr], groups[tr], featA, device)
    p1 = np.zeros(len(cand), np.float32)
    p1[tr] = oofA
    p1[~tr] = modelA.inplace_predict(XA[~tr])
    print(f"[{time.time() - t0:.0f}s] stage A done", flush=True)
    # ---- stage B
    L = list_features(cand, p1, pool_store)
    featB = featA + list(L.columns)
    X2 = np.hstack([XA, L.to_numpy(dtype=np.float32)])
    del XA, L; gc.collect()
    oofB, modelB = fit_oof(X2[tr], y[tr], groups[tr], featB, device)
    p2 = np.zeros(len(cand), np.float32)
    p2[tr] = oofB
    p2[~tr] = modelB.inplace_predict(X2[~tr])
    del X2; gc.collect()
    print(f"[{time.time() - t0:.0f}s] stage B done", flush=True)
    # ---- calibration and decoder tuning on TRAIN (out-of-fold)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oofB, y[tr])
    cand["p1"], cand["p"] = p1, iso.predict(p2).astype(np.float32)
    s1_train = all_s1[s1_bucket(all_s1) < train_pct]
    s1_dev = all_s1[(s1_bucket(all_s1) >= train_pct) & (s1_bucket(all_s1) < train_pct + dev_pct)]
    ct, cd = cand[tr], cand[~tr]
    # blocking misses per S1 (true pairs not in the candidate set) -> Poisson rate for the decoder
    gt_train = pairs[pairs.s1.isin(s1_train)]
    lam_est = (len(gt_train) - ct.y.sum()) / len(s1_train)
    best = (-1, None)
    for lam in (0.0, lam_est / 2, lam_est):
        sel = expected_f_decode(enforce_partition(ct[["s1", "m", "p"]]), lam_miss=lam)
        f = macro_f05(sel, pairs, s1_train)
        print(f"   tune lam_miss={lam:.3f}: train-OOF F0.5={f:.4f}", flush=True)
        if f > best[0]:
            best = (f, lam)
    lam = best[1]
    # ---- DEV report (held-out S1 entities)
    rep = {}
    for thr in (0.3, 0.5, 0.7):
        rep[f"stageA p1>={thr}"] = macro_f05(threshold_decode(cd, thr, "p1"), pairs, s1_dev)
    rep["stageB calibrated p>=0.5"] = macro_f05(threshold_decode(cd, 0.5), pairs, s1_dev)
    rep["+ partition constraint"] = macro_f05(threshold_decode(enforce_partition(cd[["s1", "m", "p"]]), 0.5), pairs, s1_dev)
    rep["+ expected-F0.5 decoding"] = macro_f05(expected_f_decode(enforce_partition(cd[["s1", "m", "p"]]), lam_miss=lam),
                                                pairs, s1_dev)
    gt_dev = pairs[pairs.s1.isin(s1_dev)]
    per = gt_dev.merge(cd[cd.y][["s1", "m"]], on=["s1", "m"], how="left", indicator=True)
    rep["blocking pair recall"] = float((per._merge == "both").mean())
    print("\nDEV macro F0.5 (held-out S1 entities):")
    for k, v in rep.items():
        print(f"   {k:32s} {v:.4f}")
    imp = pd.Series(modelB.get_score(importance_type="total_gain")).reindex(featB).fillna(0).sort_values(ascending=False)
    print("\nTop stage-B features by gain:\n", (imp / imp.sum()).head(20).round(4).to_string())
    pickle.dump({"modelA": modelA, "modelB": modelB, "iso": iso, "lam_miss": lam,
                 "featA": featA, "featB": featB, "report": rep},
                open(work / "models.pkl", "wb"))
    print(f"[{time.time() - t0:.0f}s] saved {work / 'models.pkl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--train-pct", type=int, default=8, help="%% of train S1 entities used to fit")
    ap.add_argument("--dev-pct", type=int, default=4, help="%% of train S1 entities held out for the report")
    a = ap.parse_args()
    main(Path(a.data), Path(a.work), a.train_pct, a.dev_pct)
