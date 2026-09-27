"""Stage 5: score test candidates, decode, and write the two submission files."""
import argparse, gc, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd

from decode import enforce_partition, expected_f_decode
from features import RecordStore, list_features, pair_features
from io_utils import write_lists


def main(work: Path, out_dir: Path, data_dir: Path, chunk_s1: int):
    """Streams one country's candidate file at a time so peak memory stays bounded (52M test pairs)."""
    t0 = time.time()
    M = pickle.load(open(work / "models.pkl", "rb"))
    s1 = pd.read_parquet(work / "n2_test_s1.parquet")
    all_s1 = s1.id.to_numpy()
    s1_store = RecordStore([s1])
    del s1
    files = sorted(work.glob("cand_test_*.parquet"))
    scored = []
    for f in files:
        cand = pd.read_parquet(f).sort_values(["s1", "src", "brank"]).reset_index(drop=True)
        need = cand.m.unique()                                   # numpy array: no Python set of millions of ints
        pool_store = RecordStore([pd.read_parquet(work / f"n2_test_s{k}.parquet").pipe(lambda d: d[d.id.isin(need)])
                                  for k in (2, 3)])
        uniq = cand.s1.unique()
        print(f"[{time.time() - t0:.0f}s] {f.stem}: {len(cand):,} pairs for {len(uniq):,} S1", flush=True)
        for i in range(0, len(uniq), chunk_s1):
            c = cand[cand.s1.isin(uniq[i:i + chunk_s1])]
            X = pair_features(c, s1_store, pool_store)[M["featA"]]
            p1 = M["modelA"].inplace_predict(X.to_numpy(dtype=np.float32))
            X2 = pd.concat([X, list_features(c, p1, pool_store)], axis=1)[M["featB"]]
            p = M["iso"].predict(M["modelB"].inplace_predict(X2.to_numpy(dtype=np.float32))).astype(np.float32)
            scored.append(pd.DataFrame({"s1": c.s1.to_numpy(), "m": c.m.to_numpy(), "p": p}))
            del X, X2, c
        print(f"[{time.time() - t0:.0f}s] scored {f.stem}", flush=True)
        del cand, pool_store
        gc.collect()
    scored = pd.concat(scored, ignore_index=True)
    scored.to_parquet(work / "scored_test.parquet")

    sel = expected_f_decode(enforce_partition(scored), lam_miss=M["lam_miss"])
    del scored
    gc.collect()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lists(out_dir / "matching_results.tsv", all_s1, sel[["s1", "m"]], "m", "matched_entity_ids")
    n_sel, n_empty = len(sel), len(all_s1) - sel.s1.nunique()
    del sel
    cand = pd.concat([pd.read_parquet(f, columns=["s1", "m"]) for f in files], ignore_index=True)
    write_lists(out_dir / "candidate_pairs.tsv", all_s1, cand, "m", "candidate_entity_ids")
    print(f"[{time.time() - t0:.0f}s] wrote outputs: {n_sel:,} matches, {n_empty:,} S1 predicted as singletons "
          f"({n_empty / len(all_s1):.1%})", flush=True)
    # Validation runs as its own step (run_pipeline.sh): the official validator alone peaks at ~7 GB.


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--chunk-s1", type=int, default=100_000)
    a = ap.parse_args()
    main(Path(a.work), Path(a.out), Path(a.data), a.chunk_s1)
