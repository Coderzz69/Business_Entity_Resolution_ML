"""Overfitting diagnostics for the two-stage matcher.

1. In-sample vs held-out: AUC, log-loss and macro F0.5 on the TRAIN S1 buckets (rows the final models
   were fit on) versus the DEV buckets (never used). A large train-dev gap means overfitting.
2. Learning curves: log-loss per boosting round on a TRAIN sample and on DEV for both stages. For an
   overfit model, DEV loss turns upward while TRAIN loss keeps falling.
3. Train/test shift: the distribution of final probabilities and matches per S1 on DEV versus TEST
   (no test labels needed).
"""
import argparse, json, pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from block import load_candidates
from decode import enforce_partition, expected_f_decode, macro_f05
from features import RecordStore, list_features, pair_features
from io_utils import load_gt
from train import s1_bucket


def score_buckets(work, M, pairs, lo, hi, max_s1=None, seed=0):
    cand = load_candidates(work, "train", s1_filter=lambda s: (s1_bucket(s) >= lo) & (s1_bucket(s) < hi))
    if max_s1:
        keep = pd.Series(cand.s1.unique()).sample(min(max_s1, cand.s1.nunique()), random_state=seed)
        cand = cand[cand.s1.isin(keep)].reset_index(drop=True)
    cand = cand.merge(pairs.assign(y=True), on=["s1", "m"], how="left")
    cand["y"] = cand.y.fillna(False).astype(bool)
    s1 = pd.read_parquet(work / "n2_train_s1.parquet")
    s1_store = RecordStore([s1[s1.id.isin(cand.s1.unique())]])
    need = cand.m.unique()
    pool = RecordStore([pd.read_parquet(work / f"n2_train_s{k}.parquet").pipe(lambda d: d[d.id.isin(need)])
                        for k in (2, 3)])
    X = pair_features(cand, s1_store, pool)[M["featA"]]
    XA = X.to_numpy(np.float32)
    p1 = M["modelA"].inplace_predict(XA)
    XB = pd.concat([X, list_features(cand, p1, pool)], axis=1)[M["featB"]].to_numpy(np.float32)
    raw = M["modelB"].inplace_predict(XB)
    cand["p1"], cand["raw"], cand["p"] = p1, raw, M["iso"].predict(raw)
    return cand, XA, XB


def summary(c, pairs, lam):
    y = c.y.to_numpy()
    sel = expected_f_decode(enforce_partition(c[["s1", "m", "p"]]), lam_miss=lam)
    return {"S1": int(c.s1.nunique()), "pairs": int(len(c)),
            "AUC_stageA": roc_auc_score(y, c.p1), "AUC_final": roc_auc_score(y, c.p),
            "logloss_stageB_raw": log_loss(y, np.clip(c.raw, 1e-7, 1 - 1e-7)),
            "macroF05": macro_f05(sel, pairs, c.s1.unique())}


def curve(booster, X, y, step=25):
    names = booster.feature_names
    n = booster.num_boosted_rounds()
    rounds = list(range(step, n + 1, step))
    d = xgb.DMatrix(X, feature_names=names)
    return rounds, [log_loss(y, np.clip(booster.predict(d, iteration_range=(0, r)), 1e-7, 1 - 1e-7)) for r in rounds]


def main(work, data_dir, out, train_pct, dev_pct, label, curve_s1):
    out.mkdir(parents=True, exist_ok=True)
    M = pickle.load(open(work / "models.pkl", "rb"))
    pairs, _ = load_gt(data_dir)
    tr, XA_tr, XB_tr = score_buckets(work, M, pairs, 0, train_pct, max_s1=60_000)
    dv, XA_dv, XB_dv = score_buckets(work, M, pairs, train_pct, train_pct + dev_pct)
    res = {"train (in-sample)": summary(tr, pairs, M["lam_miss"]), "dev (held-out)": summary(dv, pairs, M["lam_miss"])}
    rep = pd.DataFrame(res).T
    rep.loc["gap (train - dev)"] = rep.iloc[0] - rep.iloc[1]

    # learning curves on an equal-size S1 sample from each side
    rng = np.random.default_rng(0)
    curves = {}
    for name, c, XA, XB in (("train", tr, XA_tr, XB_tr), ("dev", dv, XA_dv, XB_dv)):
        s = rng.choice(c.s1.unique(), min(curve_s1, c.s1.nunique()), replace=False)
        m = c.s1.isin(s).to_numpy()
        y = c.y.to_numpy()[m]
        curves[name] = {"A": curve(M["modelA"], XA[m], y), "B": curve(M["modelB"], XB[m], y)}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, st in zip(axes[:2], ("A", "B")):
        for name, col in (("train", "#eb6834"), ("dev", "#2a78d6")):
            r, l = curves[name][st]
            ax.plot(r, l, lw=2, color=col, label=f"{name}  (final {l[-1]:.4f})")
        ax.set_title(f"Stage {st}: log-loss vs boosting round"); ax.set_xlabel("round"); ax.set_ylabel("log-loss")
        ax.set_yscale("log"); ax.legend(frameon=False); ax.grid(color="#e6e5e0")
        ax.spines[["top", "right"]].set_visible(False)

    # dev vs test shift (unlabelled test scores from predict.py)
    shift = {}
    ax = axes[2]
    bins = np.linspace(0, 1, 41)
    ax.hist(dv.p, bins=bins, density=True, histtype="step", lw=2, color="#2a78d6", label="dev")
    shift["dev"] = {"mean_p": float(dv.p.mean()), "share_p_gt_0.5": float((dv.p > 0.5).mean()),
                    "matches_per_S1": float(len(expected_f_decode(enforce_partition(dv[["s1", "m", "p"]]),
                                                                  lam_miss=M["lam_miss"])) / dv.s1.nunique())}
    tp = work / "scored_test.parquet"
    if tp.exists():
        te = pd.read_parquet(tp)
        keep = pd.Series(te.s1.unique()).sample(300_000, random_state=0)
        te = te[te.s1.isin(keep)].reset_index(drop=True)
        cty = pd.read_parquet(work / "n2_test_s1.parquet", columns=["id", "country"]).set_index("id").country
        te["country"] = cty.reindex(te.s1.to_numpy()).to_numpy()
        ax.hist(te.p, bins=bins, density=True, histtype="step", lw=2, color="#eb6834", label="test (all)")
        fr = te[te.country == "France"]
        ax.hist(fr.p, bins=bins, density=True, histtype="step", lw=2, color="#e87ba4", label="test France")
        for name, t in (("test (all)", te), ("test France", fr), ("test US+India", te[te.country != "France"])):
            sel = expected_f_decode(enforce_partition(t[["s1", "m", "p"]]), lam_miss=M["lam_miss"])
            shift[name] = {"mean_p": float(t.p.mean()), "share_p_gt_0.5": float((t.p > 0.5).mean()),
                           "matches_per_S1": float(len(sel) / t.s1.nunique())}
    ax.set_yscale("log"); ax.set_title("Final probability: dev vs test"); ax.set_xlabel("p(match)")
    ax.legend(frameon=False); ax.grid(color="#e6e5e0"); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(out / f"overfit_{label}.png", dpi=130)

    pd.set_option("display.width", 200)
    print(f"\n== {label}: in-sample vs held-out")
    print(rep.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nLearning curves (log-loss at rounds 25 / 100 / 200 / final):")
    for name in ("train", "dev"):
        for st in ("A", "B"):
            r, l = curves[name][st]
            pick = [l[r.index(k)] for k in (25, 100, 200) if k in r] + [l[-1]]
            print(f"   stage {st} {name:5s}: " + " / ".join(f"{v:.4f}" for v in pick) +
                  f"   (min at round {r[int(np.argmin(l))]})")
    print("\nDev vs test shift:")
    print(pd.DataFrame(shift).T.to_string(float_format=lambda v: f"{v:.4f}"))
    json.dump({"in_vs_out": res, "curves": curves, "shift": shift}, open(out / f"overfit_{label}.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="work/eval")
    ap.add_argument("--train-pct", type=int, default=8)
    ap.add_argument("--dev-pct", type=int, default=4)
    ap.add_argument("--label", default="full")
    ap.add_argument("--curve-s1", type=int, default=30_000)
    a = ap.parse_args()
    main(Path(a.work), Path(a.data), Path(a.out), a.train_pct, a.dev_pct, a.label, a.curve_s1)
