"""Pairwise features for (S1, candidate) pairs, plus second-stage list/coherence features.

All string similarities use rapidfuzz.process.cpdist (C++, multithreaded, element-wise), so
tens of millions of pairs are feasible on a CPU. No feature uses the country label: the model
has to transfer to France, which has no training data.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Prefix

REC_COLS = ["id", "name_tok", "name_core", "name_legal", "name_compact", "alias_core", "addr_can", "addr_num",
            "is_handle", "name_nonascii", "addr_empty"]
CTX_COLS = ["bscore", "cos_name", "cos_compact", "cos_addr", "cos_num", "brank", "b_gap_top", "b_rel", "b_second",
            "m_n_s1", "m_best", "m_is_best", "m_second", "m_margin", "src"]


class RecordStore:
    """id -> record lookup. Columns stay Arrow-backed (compact); Python strings are only created per gathered chunk."""

    def __init__(self, frames):
        self.df = pd.concat([f[REC_COLS] for f in frames], ignore_index=True)
        self.index = pd.Index(self.df.id.to_numpy())

    def gather(self, ids):
        ix = self.index.get_indexer(ids)
        if (ix < 0).any():
            raise KeyError(f"{(ix < 0).sum()} ids not found")
        sub = self.df.iloc[ix]
        return {c: (sub[c].to_numpy(dtype=object) if sub[c].dtype.kind not in "biuf" else sub[c].to_numpy())
                for c in REC_COLS if c != "id"}


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _first(tok_arr):
    return np.array([s.split(" ", 1)[0] if s else "" for s in tok_arr], dtype=object)


def _count(tok_arr):
    return np.array([s.count(" ") + 1 if s else 0 for s in tok_arr], dtype=np.int16)


def pair_features(cand: pd.DataFrame, s1_store: RecordStore, pool_store: RecordStore) -> pd.DataFrame:
    a = s1_store.gather(cand.s1.to_numpy())
    b = pool_store.gather(cand.m.to_numpy())
    f = {}
    # names
    f["n_ratio"] = _cp(a["name_core"], b["name_core"], fuzz.ratio)
    f["n_tset"] = _cp(a["name_core"], b["name_core"], fuzz.token_set_ratio)
    f["n_tsort"] = _cp(a["name_core"], b["name_core"], fuzz.token_sort_ratio)
    f["n_partial"] = _cp(a["name_core"], b["name_core"], fuzz.partial_ratio)
    f["n_comp_ratio"] = _cp(a["name_compact"], b["name_compact"], fuzz.ratio)
    f["n_comp_partial"] = _cp(a["name_compact"], b["name_compact"], fuzz.partial_ratio)
    f["n_jw"] = _cp(a["name_compact"], b["name_compact"], JaroWinkler.normalized_similarity)
    f["n_prefix"] = _cp(a["name_compact"], b["name_compact"], Prefix.similarity)
    f["n_full_tset"] = _cp(a["name_tok"], b["name_tok"], fuzz.token_set_ratio)
    f["n_alias_tset"] = _cp(a["name_core"], b["alias_core"], fuzz.token_set_ratio)
    f["legal_tset"] = _cp(a["name_legal"], b["name_legal"], fuzz.token_set_ratio)
    na, nb = _count(a["name_core"]), _count(b["name_core"])
    f["n_cnt_a"], f["n_cnt_b"], f["n_cnt_diff"] = na, nb, (nb - na).astype(np.int16)
    f["legal_empty_b"] = np.array([s == "" for s in b["name_legal"]], dtype=np.int8)
    # addresses
    f["a_tset"] = _cp(a["addr_can"], b["addr_can"], fuzz.token_set_ratio)
    f["a_tsort"] = _cp(a["addr_can"], b["addr_can"], fuzz.token_sort_ratio)
    f["a_partial"] = _cp(a["addr_can"], b["addr_can"], fuzz.partial_ratio)
    f["a_ratio"] = _cp(a["addr_can"], b["addr_can"], fuzz.ratio)
    f["num_tset"] = _cp(a["addr_num"], b["addr_num"], fuzz.token_set_ratio)
    fa, fb = _first(a["addr_num"]), _first(b["addr_num"])
    f["num_first_ratio"] = _cp(fa, fb, fuzz.ratio)
    f["num_first_eq"] = np.array([x == y and x != "" for x, y in zip(fa, fb)], dtype=np.int8)
    f["num_cnt_a"], f["num_cnt_b"] = _count(a["addr_num"]), _count(b["addr_num"])
    f["addr_empty_a"], f["addr_empty_b"] = a["addr_empty"].astype(np.int8), b["addr_empty"].astype(np.int8)
    f["b_nonascii"], f["b_handle"] = b["name_nonascii"].astype(np.int8), b["is_handle"].astype(np.int8)
    # a cross-view summary that is robust to one field being missing or garbage
    f["best_view"] = np.maximum(f["n_tset"], np.where(f["addr_empty_b"] == 1, 0, f["a_tset"]))
    out = pd.DataFrame(f, index=cand.index)
    for c in CTX_COLS:
        out[c] = cand[c].to_numpy()
    return out


# ------------------------------------------------------------------ stage-2 features

def list_features(cand: pd.DataFrame, p1: np.ndarray, pool_store: RecordStore) -> pd.DataFrame:
    """Features that look at the whole candidate list of an S1 entity, using stage-1 scores.

    * rank / gap / count statistics of p1 inside the S1 list (overall and per source);
    * cross-source coherence: the true S2 and S3 records of one entity are noisy copies of the
      same business, so a genuine candidate should also resemble the *best candidate from the
      other source*. We measure name/address/number similarity to that anchor, weighted by the
      anchor's p1. Candidates that agree with nothing else in the list get demoted.
    """
    c = cand[["s1", "m", "src"]].copy()
    c["p1"] = p1.astype(np.float32)
    g = c.groupby("s1").p1
    gs = c.groupby(["s1", "src"]).p1
    f = pd.DataFrame(index=c.index)
    f["p1"] = c.p1
    f["p1_rank"] = g.rank(ascending=False, method="first").astype(np.int16)
    f["p1_rank_src"] = gs.rank(ascending=False, method="first").astype(np.int16)
    f["p1_max"] = g.transform("max")
    f["p1_gap"] = f.p1_max - c.p1
    f["p1_sum"] = g.transform("sum")
    f["p1_n05"] = g.transform(lambda x: (x > 0.5).sum()).astype(np.int16)
    f["p1_max_src"] = gs.transform("max")
    # anchor = best candidate of the *other* source for the same S1
    best = c.loc[c.groupby(["s1", "src"]).p1.idxmax(), ["s1", "src", "m", "p1"]]
    best["src"] = np.where(best.src == 2, 3, 2).astype(np.int8)   # re-key: serves as anchor for the opposite source
    anc = c[["s1", "src"]].merge(best.rename(columns={"m": "anc_m", "p1": "anc_p1"}), on=["s1", "src"], how="left")
    has = anc.anc_m.notna().to_numpy()
    f["anc_p1"] = anc.anc_p1.fillna(0).to_numpy(np.float32)
    for name in ("anc_n_tset", "anc_a_tset", "anc_num_tset", "anc_comp_ratio"):
        f[name] = np.float32(0)
    if has.any():
        a = pool_store.gather(c.m.to_numpy()[has])
        b = pool_store.gather(anc.anc_m.to_numpy()[has].astype(np.int64))
        f.loc[has, "anc_n_tset"] = _cp(a["name_core"], b["name_core"], fuzz.token_set_ratio)
        f.loc[has, "anc_a_tset"] = _cp(a["addr_can"], b["addr_can"], fuzz.token_set_ratio)
        f.loc[has, "anc_num_tset"] = _cp(a["addr_num"], b["addr_num"], fuzz.token_set_ratio)
        f.loc[has, "anc_comp_ratio"] = _cp(a["name_compact"], b["name_compact"], fuzz.ratio)
    f["anc_support"] = f.anc_p1 * np.maximum(f.anc_n_tset, f.anc_a_tset) / 100
    return f
