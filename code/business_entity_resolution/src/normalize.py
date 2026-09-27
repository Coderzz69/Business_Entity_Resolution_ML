"""Country-agnostic text normalisation for business names and addresses.

Pass 1 (`normalize_frame`) is purely rule-based: transliterate any script to ASCII,
lowercase, split aliases (a/k/a, dba, f/k/a ...), strip web-handle / domain wrappers,
and tokenise. Pass 2 (`apply_synonyms`) maps tokens through the synonym table that
`synonyms.py` learns from matched training pairs (st->street, praivet->private,
tx->texas, ...), then splits names into core and legal tokens.
"""
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from anyascii import anyascii

N_JOBS = 12

ALIAS_RE = re.compile(r"\b(?:a/k/a|aka|d/b/a|dba|f/k/a|fka|formerly(?:\s+known\s+as)?|t/a|trading\s+as|doing\s+business\s+as)\b", re.I)
DOMAIN_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|in|co\.in|net|org|fr|biz|info|co|us)\b")
NONALNUM_RE = re.compile(r"[^a-z0-9]+")
DIGITS_RE = re.compile(r"\d+")

# Seed abbreviations (general orthography, not entity data). Learned synonyms extend these.
SEED_ADDR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "cir": "circle",
    "hwy": "highway", "pkwy": "parkway", "sq": "square", "ter": "terrace", "trl": "trail", "mt": "mount",
    "ste": "suite", "apt": "apartment", "fl": "floor", "bldg": "building", "r": "rue", "ch": "chemin",
    "rte": "route", "imp": "impasse", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}
SEED_NAME = {
    "pvt": "private", "ltd": "limited", "inc": "incorporated", "corp": "corporation", "co": "company",
    "cie": "compagnie", "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "service",
    "assoc": "associates", "bros": "brothers",
}
# Words that describe legal form / filler rather than identity. Kept separately for a suffix-agreement feature.
LEGAL = frozenset("""private limited pvt ltd inc corp co llp llc incorporated corporation company pc lp plc pllc ltda gmbh
    sarl sas sasu sa eurl sci snc ei scop scp selarl cie compagnie the and of et de du des la le les
    group services service trust foundation holdings enterprises industries international center
    centre sri shri smt""".split())
ADDR_FILLER = frozenset("""h no hno d door unit apartment flat plot pmb box po floor suite building
    near opp opposite behind beside next to and the of""".split())


def _translit(s: str) -> str:
    return anyascii(s) if not s.isascii() else s


def _norm_number(tok: str) -> str:
    t = tok.lstrip("0")
    return t if t else "0"


def _name_one(raw: str):
    s = _translit(raw).lower()
    is_handle = int(bool(re.search(r"^\s*[@#]|\.(?:com|in|net|org|fr)\b|www\.", s)))
    s = DOMAIN_RE.sub(r" \1 ", s)
    parts = ALIAS_RE.split(s)
    main = parts[-1] if len(parts) > 1 and parts[-1].strip() else s
    alias = parts[0] if len(parts) > 1 else ""
    toks = [t for t in NONALNUM_RE.split(main.replace("&", " and ")) if t]
    # Drop long digit runs (phone numbers appended as noise)
    toks = [t for t in toks if not (t.isdigit() and len(t) >= 7)]
    return " ".join(toks), NONALNUM_RE.sub(" ", alias).strip(), is_handle


def _addr_one(raw: str):
    s = _translit(raw).lower().replace("n°", " no ").replace("#", " ")
    s = re.sub(r"\bnull\b", " ", s)
    toks = [t for t in NONALNUM_RE.split(s) if t]
    out, nums = [], []
    for t in toks:
        if t.isdigit():
            t = _norm_number(t)
            nums.append(t)
        elif any(c.isdigit() for c in t):
            # 43945c / 2nd / b204: keep the number, plus the token itself
            for d in DIGITS_RE.findall(t):
                nums.append(_norm_number(d))
        out.append(t)
    return " ".join(out), " ".join(nums)


def _chunk(args):
    names, addrs = args
    n = [_name_one(x) for x in names]
    a = [_addr_one(x) for x in addrs]
    return n, a


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """df has columns id, name, address, country. Returns pass-1 normalised columns."""
    names = df["name"].to_numpy(dtype=object)
    addrs = df["address"].to_numpy(dtype=object)
    step = max(1, len(df) // (N_JOBS * 8) + 1)
    jobs = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(df), step)]
    with ProcessPoolExecutor(N_JOBS) as ex:
        res = list(ex.map(_chunk, jobs))
    n = [x for r in res for x in r[0]]
    a = [x for r in res for x in r[1]]
    out = pd.DataFrame({
        "id": df["id"].to_numpy(),
        "country": df["country"].to_numpy(dtype=object),
        "name_tok": [x[0] for x in n],
        "alias_tok": [x[1] for x in n],
        "is_handle": np.array([x[2] for x in n], dtype=np.int8),
        "addr_tok": [x[0] for x in a],
        "addr_num": [x[1] for x in a],
        "name_nonascii": ~df["name"].str.isascii().to_numpy(dtype=bool),
        "addr_empty": (df["address"].str.strip() == "").to_numpy(dtype=bool),
    })
    return out


# ---------------------------------------------------------------- pass 2

def _map_tokens(s: str, syn: dict) -> list:
    return [syn.get(t, t) for t in s.split()]


def _segment(tok: str, vocab: dict, max_len=20):
    """Unigram-LM word segmentation of a glued handle ('maabigconsultancy' -> maa big consultancy)."""
    n = len(tok)
    best = [(0.0, -1)] + [(-1e18, -1)] * n
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            w = tok[j:i]
            lp = vocab.get(w)
            if lp is None:
                continue
            sc = best[j][0] + lp
            if sc > best[i][0]:
                best[i] = (sc, j)
    if best[n][1] < 0:
        return None
    words, i = [], n
    while i > 0:
        j = best[i][1]
        words.append(tok[j:i]); i = j
    words = words[::-1]
    return words if len(words) > 1 else None


_T = {}


def _init_pass2(tables, vocab):
    _T.update(tables); _T["vocab"] = vocab


def _apply_phrases(toks, phr):
    if not phr or len(toks) < 2:
        return toks
    out, i = [], 0
    while i < len(toks):
        if i + 1 < len(toks):
            m = phr.get(toks[i] + " " + toks[i + 1])
            if m is not None:
                out.extend(m.split()); i += 2; continue
        out.append(toks[i]); i += 1
    return out


def _canon_name(s: str, split_handles: bool):
    syn, phr, vocab = _T["name"], _T["name_phrase"], _T["vocab"]
    toks = []
    for t in s.split():
        if split_handles and len(t) >= 7 and t not in vocab and not t.isdigit():
            seg = _segment(t, vocab)
            if seg:
                toks.extend(seg); continue
        toks.append(t)
    toks = _apply_phrases(toks, phr)
    return [syn.get(t, t) for t in toks]


def _pass2_chunk(args):
    name_tok, alias_tok, addr_tok = args
    core, legal, compact, alias_core, addr = [], [], [], [], []
    syn_a, phr_a = _T["addr"], _T["addr_phrase"]
    for nt, at, ad in zip(name_tok, alias_tok, addr_tok):
        toks = _canon_name(nt, True)
        c = [t for t in toks if t not in LEGAL]
        core.append(" ".join(c))
        legal.append(" ".join(sorted(set(t for t in toks if t in LEGAL))))
        compact.append("".join(c))
        alias_core.append(" ".join(t for t in _canon_name(at, False) if t not in LEGAL))
        addr.append(" ".join(syn_a.get(t, t) for t in _apply_phrases(ad.split(), phr_a)))
    return core, legal, compact, alias_core, addr


def apply_synonyms(df: pd.DataFrame, tables: dict, vocab: dict) -> pd.DataFrame:
    """Pass 2: learned synonyms + handle segmentation. Adds name_core/name_legal/name_compact/alias_core/addr_can."""
    step = max(1, len(df) // (N_JOBS * 8) + 1)
    cols = [df[c].to_numpy(dtype=object) for c in ("name_tok", "alias_tok", "addr_tok")]
    jobs = [(cols[0][i:i + step], cols[1][i:i + step], cols[2][i:i + step]) for i in range(0, len(df), step)]
    tabs = {k: tables[k] for k in ("name", "name_phrase", "addr", "addr_phrase")}
    with ProcessPoolExecutor(N_JOBS, initializer=_init_pass2, initargs=(tabs, vocab)) as ex:
        res = list(ex.map(_pass2_chunk, jobs))
    out = df.copy()
    for k, name in enumerate(["name_core", "name_legal", "name_compact", "alias_core", "addr_can"]):
        out[name] = [x for r in res for x in r[k]]
    return out
