"""Discrimination metrics on held-out S1 entities: ROC AUC and Somers' D.

Re-scores the DEV candidate pairs (same hash split as train.py, never used for fitting) with the saved
models, then reports, for the blocking score, stage A and stage B (calibrated):
  * ROC AUC over candidate pairs (label = true match);
  * Somers' D(score|outcome) = 2*AUC - 1 for a binary outcome (checked against scipy.stats.somersd on a
    subsample);
  * the same, broken down by country and by source (S2/S3).
Pairs that blocking never proposed are outside this evaluation; blocking recall is reported beside it.
"""
import argparse, json, pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import somersd
from sklearn.metrics import roc_auc_score, roc_curve

from block import load_candidates
from features import RecordStore, list_features, pair_features
from io_utils import load_gt
from train import s1_bucket

SCORES = {"bscore": "Blocking cosine", "p1": "Stage A (pairwise)", "p": "Stage B (calibrated, final)"}
COLORS = {"bscore": "#b8b7b0", "p1": "#eb6834", "p": "#2a78d6"}


def score_dev(work: Path, data_dir: Path, train_pct: int, dev_pct: int, chunk=1_500_000) -> pd.DataFrame:
    M = pickle.load(open(work / "models.pkl", "rb"))
    b = lambda s: s1_bucket(s)
    cand = load_candidates(work, "train", s1_filter=lambda s: (b(s) >= train_pct) & (b(s) < train_pct + dev_pct))
    pairs, _ = load_gt(data_dir)
    cand = cand.merge(pairs.assign(y=True), on=["s1", "m"], how="left")
    cand["y"] = cand.y.fillna(False).astype(bool)
    s1 = pd.read_parquet(work / "n2_train_s1.parquet")
    s1 = s1[s1.id.isin(cand.s1.unique())]
    country = s1.set_index("id").country
    s1_store = RecordStore([s1])
    need = cand.m.unique()
    pool_store = RecordStore([pd.read_parquet(work / f"n2_train_s{k}.parquet").pipe(lambda d: d[d.id.isin(need)])
                              for k in (2, 3)])
    uniq = cand.s1.unique()
    out = []
    step = max(1, int(len(uniq) * chunk / max(len(cand), 1)))
    for i in range(0, len(uniq), step):
        c = cand[cand.s1.isin(uniq[i:i + step])]
        X = pair_features(c, s1_store, pool_store)[M["featA"]]
        p1 = M["modelA"].inplace_predict(X.to_numpy(dtype=np.float32))
        X2 = pd.concat([X, list_features(c, p1, pool_store)], axis=1)[M["featB"]]
        p = M["iso"].predict(M["modelB"].inplace_predict(X2.to_numpy(dtype=np.float32)))
        out.append(pd.DataFrame({"s1": c.s1.to_numpy(), "src": c.src.to_numpy(), "y": c.y.to_numpy(),
                                 "bscore": c.bscore.to_numpy(), "p1": p1.astype(np.float32),
                                 "p": p.astype(np.float32)}))
    d = pd.concat(out, ignore_index=True)
    d["country"] = country.reindex(d.s1.to_numpy()).to_numpy()
    gt_dev = pairs[pairs.s1.isin(uniq)]
    d.attrs["blocking_recall"] = float(d.y.sum() / len(gt_dev))
    return d


def metrics(y, s, n_check=20_000, seed=0):
    auc = roc_auc_score(y, s)
    res = {"n_pairs": int(len(y)), "positives": int(y.sum()), "AUC": auc, "SomersD": 2 * auc - 1}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), min(n_check, len(y)), replace=False)
    # scipy.stats.somersd(x, y) returns D(y|x). The AUC identity holds for D(score|outcome):
    # pairs are split by outcome and we count how often the match has the higher score.
    res["SomersD_scipy_subsample"] = float(somersd(y[idx].astype(int), s[idx]).statistic)
    res["AUC_subsample"] = float(roc_auc_score(y[idx], s[idx]))
    return res


def main(work: Path, data_dir: Path, out: Path, train_pct: int, dev_pct: int, label: str):
    out.mkdir(parents=True, exist_ok=True)
    d = score_dev(work, data_dir, train_pct, dev_pct)
    y = d.y.to_numpy()
    rows = []
    for col, name in SCORES.items():
        rows.append({"slice": "all", "score": name, **metrics(y, d[col].to_numpy())})
    for key, grp in list(d.groupby("country")) + [(f"S{k}", g) for k, g in d.groupby("src")]:
        rows.append({"slice": key, "score": SCORES["p"], **metrics(grp.y.to_numpy(), grp.p.to_numpy())})
    rep = pd.DataFrame(rows)
    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, zoom in zip(axes, (False, True)):
        for col, name in SCORES.items():
            fpr, tpr, _ = roc_curve(y, d[col].to_numpy())
            auc = rep.loc[(rep.slice == "all") & (rep.score == name), "AUC"].iloc[0]
            ax.plot(fpr, tpr, lw=2, color=COLORS[col], label=f"{name}  AUC={auc:.4f}  D={2 * auc - 1:.4f}")
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#8a8983")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        if zoom:
            ax.set_xlim(0, 0.05); ax.set_ylim(0.8, 1.0); ax.set_title("Zoom: FPR ≤ 5%")
        else:
            ax.set_title(f"ROC on held-out candidate pairs ({label})")
        ax.grid(color="#e6e5e0"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(out / f"roc_{label}.png", dpi=130)
    rep.to_csv(out / f"roc_somersd_{label}.csv", index=False)
    json.dump({"label": label, "blocking_recall": d.attrs["blocking_recall"], "rows": rows},
              open(out / f"roc_somersd_{label}.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 200)
    print(f"\n== {label}: {d.s1.nunique():,} held-out S1, {len(d):,} candidate pairs, "
          f"blocking recall {d.attrs['blocking_recall']:.4f}")
    print(rep[["slice", "score", "n_pairs", "positives", "AUC", "SomersD", "SomersD_scipy_subsample"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output/eval")
    ap.add_argument("--train-pct", type=int, default=8)
    ap.add_argument("--dev-pct", type=int, default=4)
    ap.add_argument("--label", default="full")
    a = ap.parse_args()
    main(Path(a.work), Path(a.data), Path(a.out), a.train_pct, a.dev_pct, a.label)
