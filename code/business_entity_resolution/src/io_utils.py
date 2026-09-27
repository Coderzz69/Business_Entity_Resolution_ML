"""Reading raw TSVs, caching them as parquet, and writing submission files."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
SRC_MUL = 10**10


def id_to_int(s: pd.Series) -> np.ndarray:
    """'S2-12345' -> 2*1e10 + 12345 (int64). Cheap, collision-free id encoding."""
    s = s.astype(str)
    return s.str.slice(1, 2).astype("int64").to_numpy() * SRC_MUL + s.str.slice(3).astype("int64").to_numpy()


def int_to_id(a) -> pd.Series:
    a = pd.Series(np.asarray(a, dtype=np.int64))
    return "S" + (a // SRC_MUL).astype(str) + "-" + (a % SRC_MUL).astype(str)


def raw_path(data_dir: Path, split: str, src: int) -> Path:
    return Path(data_dir) / split / f"{split}_source{src}.tsv"


def cache_source(data_dir: Path, work: Path, split: str, src: int) -> Path:
    """TSV -> parquet (id int64, name, address, country). Returns parquet path."""
    out = Path(work) / f"raw_{split}_s{src}.parquet"
    if out.exists():
        return out
    df = pd.read_csv(raw_path(data_dir, split, src), **READ_KW)
    tbl = pa.table({
        "id": id_to_int(df.entity_id),
        "name": df.business_name.to_numpy(dtype=object),
        "address": df.business_address.to_numpy(dtype=object),
        "country": df.country.to_numpy(dtype=object),
    })
    pq.write_table(tbl, out)
    return out


def load_gt(data_dir: Path) -> pd.DataFrame:
    """Ground truth as a long table of (s1, m) int pairs plus the list of all S1 ids."""
    gt = pd.read_csv(Path(data_dir) / "train" / "train_ground_truth.tsv", **READ_KW)
    s1 = id_to_int(gt.source1_entity_id)
    has = gt.matched_entity_ids != ""
    exp = gt.loc[has, ["source1_entity_id", "matched_entity_ids"]]
    exp = exp.assign(m=exp.matched_entity_ids.str.split(",")).explode("m")
    pairs = pd.DataFrame({"s1": id_to_int(exp.source1_entity_id), "m": id_to_int(exp.m)})
    return pairs, s1


def write_lists(path: Path, s1_ids: np.ndarray, pairs: pd.DataFrame, col: str, header: str, chunk=200_000):
    """Write one row per S1 id; `pairs` has int columns s1, m. S1 ids without pairs get an empty list.
    Streams in S1 chunks so tens of millions of IDs never exist as strings at once."""
    s1_ids = np.asarray(s1_ids)
    pairs = pairs[["s1", "m"]].drop_duplicates().sort_values(["s1", "m"])
    ps1 = pairs.s1.to_numpy()
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{header}\n")
        for i in range(0, len(s1_ids), chunk):
            ids = s1_ids[i:i + chunk]
            sub = pairs[np.isin(ps1, ids)]
            lists = pd.Series(int_to_id(sub.m.to_numpy()).to_numpy(), index=sub.s1.to_numpy()).groupby(level=0).agg(",".join)
            vals = lists.reindex(ids).fillna("").to_numpy()
            for a, b in zip(int_to_id(ids).to_numpy(), vals):
                fh.write(f"{a}\t{b}\n")


def load_env(path=".env") -> dict:
    """Minimal KEY=VALUE reader for the project .env (no external dependency)."""
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env
