"""Learn token-substitution tables from matched training pairs.

For each true pair (S1 record, S2/S3 record) we take the residual tokens that appear on
only one side, in their original order, and align them:
  * equal-length residuals are aligned by position (transliterated names keep word order:
    'globl kmstrksms praivet limited' <-> 'global constructions private limited');
  * a 1-token vs 2-adjacent-token residual yields a phrase rule ('new york' <-> 'ny').
Substitutions that recur often, in both directions, are merged with union-find, and
each class maps to its most frequent member. Names and addresses get separate tables.
"""
import math
from collections import Counter

import pandas as pd

from normalize import SEED_ADDR, SEED_NAME


def _ordered_residual(toks, other):
    seen, out = set(), []
    for t in toks:
        if t not in other and t not in seen and not any(ch.isdigit() for ch in t):
            seen.add(t); out.append(t)
    return out


def _adjacent(toks, pair):
    for i in range(len(toks) - 1):
        if toks[i] == pair[0] and toks[i + 1] == pair[1]:
            return True
    return False


def _align_counts(a_col, b_col):
    pair_c, a_c, b_c = Counter(), Counter(), Counter()
    for a, b in zip(a_col, b_col):
        ta, tb = a.split(), b.split()
        sa, sb = set(ta), set(tb)
        ra, rb = _ordered_residual(ta, sb), _ordered_residual(tb, sa)
        a_c.update(ra); b_c.update(rb)
        if not ra or not rb:
            continue
        if len(ra) == len(rb):
            for x, y in zip(ra, rb):
                pair_c[(x, y)] += 1
        elif len(ra) == 1 and len(rb) == 2 and _adjacent(tb, rb):
            pair_c[(ra[0], " ".join(rb))] += 1
            b_c[" ".join(rb)] += 1
        elif len(ra) == 2 and len(rb) == 1 and _adjacent(ta, ra):
            pair_c[(" ".join(ra), rb[0])] += 1
            a_c[" ".join(ra)] += 1
    return pair_c, a_c, b_c


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def learn_table(a_col, b_col, freq: Counter, min_pair=15, p_fwd=0.35, p_back=0.03, max_class=8):
    """Returns (unigram_map, phrase_map, edges). phrase_map keys are 'w1 w2' strings."""
    pair_c, a_c, b_c = _align_counts(a_col, b_col)
    uf, edges = _UF(), []
    for (x, y), c in pair_c.items():
        if c < min_pair or x == y:
            continue
        if c / max(b_c[y], 1) >= p_fwd and c / max(a_c[x], 1) >= p_back:
            edges.append((x, y, c))
            uf.union(x, y)
    classes = {}
    for t in list(uf.p):
        classes.setdefault(uf.find(t), []).append(t)
    phrase_freq = Counter()
    for (x, y), c in pair_c.items():
        for t in (x, y):
            if " " in t:
                phrase_freq[t] += c
    uni, phr = {}, {}
    for members in classes.values():
        if len(members) > max_class:
            continue
        # Canonical form: prefer single tokens (cheaper to match), then the most frequent
        canon = max(members, key=lambda t: (" " not in t, freq.get(t, 0) + phrase_freq.get(t, 0)))
        for m in members:
            if m == canon:
                continue
            (phr if " " in m else uni)[m] = canon
    return uni, phr, edges


def merge_seed(table: dict, seed: dict) -> dict:
    out = dict(table)
    for k, v in seed.items():
        if k != v and k not in out:
            out[k] = out.get(v, v)
    for k in list(out):
        seen, v = {k}, out[k]
        while v in out and v not in seen:
            seen.add(v); v = out[v]
        out[k] = v
    return {k: v for k, v in out.items() if k != v}


def build_vocab(token_series_list, min_count=5):
    c = Counter()
    for s in token_series_list:
        for x in s:
            c.update(x.split())
    total = sum(c.values())
    vocab = {w: math.log(n / total) for w, n in c.items() if n >= min_count and len(w) >= 2 and not w.isdigit()}
    return vocab, c


def learn_all(pairs: pd.DataFrame, s1: pd.DataFrame, oth: pd.DataFrame, freq_name: Counter, freq_addr: Counter,
              n_pairs=2_000_000, seed=0):
    """pairs: (s1, m) ints. s1/oth: pass-1 frames indexed by id. Returns a dict of tables."""
    p = pairs.sample(min(n_pairs, len(pairs)), random_state=seed)
    a = s1.reindex(p.s1.to_numpy())
    b = oth.reindex(p.m.to_numpy())
    ok = a.name_tok.notna().to_numpy() & b.name_tok.notna().to_numpy()
    a, b = a[ok], b[ok]
    n_uni, n_phr, n_edges = learn_table(a.name_tok.to_numpy(), b.name_tok.to_numpy(), freq_name)
    a_uni, a_phr, a_edges = learn_table(a.addr_tok.to_numpy(), b.addr_tok.to_numpy(), freq_addr)
    return {"name": merge_seed(n_uni, SEED_NAME), "name_phrase": n_phr, "addr": merge_seed(a_uni, SEED_ADDR), "addr_phrase": a_phr,
            "name_edges": n_edges, "addr_edges": a_edges}
