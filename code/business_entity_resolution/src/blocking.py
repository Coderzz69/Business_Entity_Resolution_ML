"""Candidate generation: IDF-weighted cosine over unigram + conjunction keys, top-K via sparse products.

Unigram keys per record:
  n_<tok>  canonical core-name tokens       q_<pre>  4-char prefix of long name tokens (typo tolerance)
  a_<tok>  canonical address words          d_<num>  normalised house/plot/postal numbers
  c_<str>  glued core name (web handles that segmentation could not split)
Conjunction keys (rare and precise, so they survive aggressive frequency pruning):
  nn_  pairs of name tokens      nd_  name token x number      na_  name token x rare address word
  da_  number x rare address word
"Rare address words" are the record's two address words with the lowest document frequency in the
pool, usually the street or locality rather than city/state. Keys are hashed (no vocabulary dict),
weighted by the pool IDF, and keys with document frequency > max_df are dropped from the search index
only. For every S1 record the top-K S2 and top-K S3 records by cosine are kept. Blocking is per country,
never across countries, and country-agnostic, so France needs no labels.

Per-view cosines (name / address / numbers / compact), computed on unigram keys with no pruning, are also
returned as matcher features.
"""
import time
from collections import Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn

N_THREADS = 12
N_FEATURES = 2**24
VIEWS = {"n_": "cos_name", "c_": "cos_compact", "a_": "cos_addr", "d_": "cos_num"}


def _hv():
    return HashingVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, n_features=N_FEATURES,
                             alternate_sign=False, norm=None, binary=True, dtype=np.float32)


def _parse(df):
    return [(core.split(), [t for t in addr.split() if not t.isdigit()], list(dict.fromkeys(num.split())), comp)
            for core, addr, num, comp in zip(df.name_core.to_numpy(), df.addr_can.to_numpy(),
                                              df.addr_num.to_numpy(), df.name_compact.to_numpy())]


def _docs(parsed, addr_df: Counter):
    out = []
    for nt, aw, nums, comp in parsed:
        toks = ["n_" + t for t in nt]
        toks += ["q_" + t[:4] for t in nt if len(t) >= 6 and not t.isdigit()]
        toks += ["a_" + t for t in aw]
        toks += ["d_" + t for t in nums]
        if len(comp) >= 6:
            toks.append("c_" + comp)
        n3, n2, num2 = nt[:3], nt[:2], nums[:2]
        rare = sorted(set(aw), key=lambda w: addr_df.get(w, 0))[:2]
        toks += ["nn_" + x + "_" + y for i, x in enumerate(n3) for y in n3[i + 1:]]
        toks += ["nd_" + x + "_" + d for x in n2 for d in num2]
        toks += ["na_" + x + "_" + w for x in n2 for w in rare]
        toks += ["da_" + d + "_" + w for d in num2 for w in rare]
        out.append(" ".join(toks))
    return out


def _view_docs(parsed, prefix):
    out = []
    for nt, aw, nums, comp in parsed:
        if prefix == "n_":
            out.append(" ".join(nt))
        elif prefix == "a_":
            out.append(" ".join(aw))
        elif prefix == "d_":
            out.append(" ".join(nums))
        else:
            out.append(comp)
    return out


def _topk(A, BT, k, chunk=100_000):
    rows, cols, vals = [], [], []
    for i in range(0, A.shape[0], chunk):
        C = sp_matmul_topn(A[i:i + chunk], BT, top_n=k, threshold=1e-6, sort=True, n_threads=N_THREADS).tocoo()
        rows.append(C.row + i); cols.append(C.col); vals.append(C.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32)


def _rowdot(A, B, r, c, chunk=2_000_000):
    out = np.empty(len(r), dtype=np.float32)
    for i in range(0, len(r), chunk):
        out[i:i + chunk] = np.asarray(A[r[i:i + chunk]].multiply(B[c[i:i + chunk]]).sum(axis=1)).ravel()
    return out


CHUNK = 250_000
N_PROC = 6
PARSE_COLS = ["name_core", "addr_can", "addr_num", "name_compact"]
_G = {}


def _init(addr_df):
    _G["addr_df"] = addr_df


def _hash_task(args):
    frame, mode = args
    parsed = _parse(frame)
    docs = _docs(parsed, _G["addr_df"]) if mode == "all" else _view_docs(parsed, mode)
    return _hv().transform(docs).tocsr()


def _addr_task(frame):
    c = Counter()
    for _, aw, _, _ in _parse(frame):
        c.update(set(aw))
    return c


def _chunks(frame):
    return [frame.iloc[i:i + CHUNK][PARSE_COLS] for i in range(0, len(frame), CHUNK)]


def _hashed(ex, frame, mode):
    return sp.vstack(list(ex.map(_hash_task, [(c, mode) for c in _chunks(frame)])), format="csr")


def _weight_norm_(M, idf):
    """In place: multiply by IDF, then L2-normalise the rows."""
    M.data *= idf[M.indices]
    l2norm(M, copy=False)
    return M


def _prune_(M, drop):
    M.data[drop[M.indices]] = 0
    M.eliminate_zeros()
    return M


def _addr_df(frames):
    total = Counter()
    with ProcessPoolExecutor(N_PROC, mp_context=mp.get_context("fork")) as ex:
        for c in ex.map(_addr_task, [c for f in frames for c in _chunks(f)]):
            total.update(c)
    return total


def _idf(df, n):
    return (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)


def block_country(s1: pd.DataFrame, pools: dict, k: int, max_df: int = 5000, verbose=True) -> pd.DataFrame:
    """s1: canonical S1 frame (one country). pools: {2: frame, 3: frame}. Returns the candidate frame."""
    t = time.time()
    srcs = list(pools)
    n_pool = sum(len(p) for p in pools.values())
    addr_df = _addr_df(pools.values())
    ex = ProcessPoolExecutor(N_PROC, mp_context=mp.get_context("fork"), initializer=_init, initargs=(addr_df,))
    Bs = {s: _hashed(ex, pools[s], "all") for s in srcs}
    df = sum(np.bincount(Bs[s].indices, minlength=N_FEATURES) for s in srcs).astype(np.float32)
    idf, drop = _idf(df, n_pool), df > max_df
    A = _prune_(_weight_norm_(_hashed(ex, s1, "all"), idf), drop)
    out = []
    for s in srcs:
        BT = _prune_(_weight_norm_(Bs.pop(s), idf), drop).T.tocsr()
        r, c, v = _topk(A, BT, k)
        del BT
        out.append(pd.DataFrame({"s1": s1.id.to_numpy()[r], "m": pools[s].id.to_numpy()[c], "bscore": v,
                                 "_r": r, "_c": c, "_s": np.int8(s)}))
    del A
    res = pd.concat(out, ignore_index=True)
    # Per-view IDF cosines on unpruned unigram keys
    for pre, name in VIEWS.items():
        Bv = {s: _hashed(ex, pools[s], pre) for s in srcs}
        dfv = sum(np.bincount(Bv[s].indices, minlength=N_FEATURES) for s in srcs).astype(np.float32)
        idfv = _idf(dfv, n_pool)
        Av = _weight_norm_(_hashed(ex, s1, pre), idfv)
        vals = np.zeros(len(res), dtype=np.float32)
        for s in srcs:
            msk = (res._s == s).to_numpy()
            vals[msk] = _rowdot(Av, _weight_norm_(Bv.pop(s), idfv), res._r.to_numpy()[msk], res._c.to_numpy()[msk])
        res[name] = vals
    ex.shutdown()
    res = res.drop(columns=["_r", "_c", "_s"])
    if verbose:
        print(f"   blocked {len(s1):,} S1 x {n_pool:,} pool -> {len(res):,} pairs in {time.time() - t:.0f}s", flush=True)
    return res
