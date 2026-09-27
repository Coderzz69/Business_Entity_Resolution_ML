"""Stage 3: candidate generation for a split, plus reverse-competition context features.

Reverse competition uses the partition constraint found in the EDA: every S2/S3 record
belongs to at most one S1 entity. For each candidate record m we therefore record how
strongly it is claimed by *other* S1 records (its best blocking score anywhere, whether
this S1 is its best claimant, the margin, and how many S1 lists contain it).
"""
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import block_country

COLS = ["id", "country", "name_core", "addr_can", "addr_num", "name_compact"]


def _top2(keys: pd.Series, vals: pd.Series):
    """Per-row: max and second-largest of `vals` within the row's key group, plus group size."""
    order = np.lexsort((-vals.to_numpy(), keys.to_numpy()))
    k, v = keys.to_numpy()[order], vals.to_numpy()[order]
    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    first_idx = np.flatnonzero(start)
    size = np.diff(np.r_[first_idx, len(k)])
    top = v[first_idx]
    second = np.where(size > 1, v[np.minimum(first_idx + 1, len(v) - 1)], 0.0)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    g = gid[inv]
    return top[g].astype(np.float32), second[g].astype(np.float32), size[g]


def add_context(c: pd.DataFrame) -> pd.DataFrame:
    c["src"] = (c.m // 10**10).astype(np.int8)
    key = c.s1 * 4 + c.src
    c["brank"] = c.groupby(key.to_numpy()).bscore.rank(ascending=False, method="first").astype(np.int16)
    top, second, _ = _top2(key, c.bscore)
    c["b_gap_top"] = top - c.bscore
    c["b_rel"] = c.bscore / np.maximum(top, 1e-6)
    c["b_second"] = second
    m_top, m_second, m_n = _top2(c.m, c.bscore)
    c["m_n_s1"] = m_n.astype(np.int16)
    c["m_best"] = m_top
    c["m_is_best"] = (c.bscore >= m_top - 1e-7).astype(np.int8)
    c["m_second"] = m_second
    c["m_margin"] = np.where(c.m_is_best == 1, c.bscore - m_second, c.bscore - m_top).astype(np.float32)
    return c


def run(work: Path, split: str, k: int, max_df: int, s1_frac: float = 1.0, seed: int = 0):
    """Writes one candidate file per country: cand_<split>_<country>.parquet (S2/S3 records never cross countries)."""
    s1 = pd.read_parquet(work / f"n2_{split}_s1.parquet", columns=COLS)
    if s1_frac < 1:
        s1 = s1.sample(frac=s1_frac, random_state=seed)
    np.save(work / f"s1ids_{split}.npy", s1.id.to_numpy())
    for old in work.glob(f"cand_{split}_*.parquet"):
        old.unlink()
    pools = {k_: pd.read_parquet(work / f"n2_{split}_s{k_}.parquet", columns=COLS) for k_ in (2, 3)}
    total = 0
    for country in sorted(s1.country.unique()):
        t = time.time()
        p = {k_: v[v.country == country].reset_index(drop=True) for k_, v in pools.items()}
        c = add_context(block_country(s1[s1.country == country].reset_index(drop=True), p, k, max_df))
        c.to_parquet(work / f"cand_{split}_{country}.parquet")
        total += len(c)
        print(f"  {split}/{country}: {len(c):,} pairs in {time.time() - t:.0f}s", flush=True)
        del c, p
    print(f"{split}: {total:,} candidate pairs", flush=True)


def load_candidates(work: Path, split: str, columns=None, s1_filter=None) -> pd.DataFrame:
    parts = []
    for f in sorted(work.glob(f"cand_{split}_*.parquet")):
        c = pd.read_parquet(f, columns=columns)
        if s1_filter is not None:
            c = c[s1_filter(c.s1.to_numpy())]
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max-df", type=int, default=5000)
    ap.add_argument("--s1-frac", type=float, default=1.0)
    a = ap.parse_args()
    run(Path(a.work), a.split, a.k, a.max_df, a.s1_frac)
