"""Stage 2: learn synonym tables from train ground truth, then canonicalise all sources."""
import argparse, pickle, time
from pathlib import Path

import pandas as pd

from io_utils import load_gt
from normalize import apply_synonyms
from synonyms import build_vocab, learn_all


def main(data_dir: Path, work: Path):
    tab_path = work / "tables.pkl"
    if not tab_path.exists():
        t = time.time()
        pairs, _ = load_gt(data_dir)
        s1 = pd.read_parquet(work / "n1_train_s1.parquet").set_index("id")
        oth = pd.concat([pd.read_parquet(work / f"n1_train_s{k}.parquet") for k in (2, 3)]).set_index("id")
        # Segmentation vocabulary uses name text from every file (no labels involved)
        extra = [pd.read_parquet(work / f"n1_test_s{k}.parquet", columns=["name_tok"]).name_tok for k in (1, 2, 3)]
        vocab, freq_name = build_vocab([s1.name_tok, oth.name_tok] + extra)
        _, freq_addr = build_vocab([s1.addr_tok, oth.addr_tok], 1)
        tables = learn_all(pairs, s1, oth, freq_name, freq_addr)
        pickle.dump((tables, vocab), open(tab_path, "wb"))
        print(f"learned tables in {time.time() - t:.0f}s: name={len(tables['name'])} addr={len(tables['addr'])}", flush=True)
        del s1, oth, pairs
    tables, vocab = pickle.load(open(tab_path, "rb"))
    for split in ("train", "test"):
        for src in (1, 2, 3):
            out = work / f"n2_{split}_s{src}.parquet"
            if out.exists():
                continue
            t = time.time()
            df = pd.read_parquet(work / f"n1_{split}_s{src}.parquet")
            apply_synonyms(df, tables, vocab).to_parquet(out)
            print(f"canonicalised {split} S{src} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    main(Path(a.data), Path(a.work))
