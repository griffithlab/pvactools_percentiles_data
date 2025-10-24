#!/usr/bin/env python3
"""Build HDF5 percentile reference files from .tsv(.gz) score files.

Usage: python build_percentiles_hdf5.py /path/to/input_folder --out-dir /path/to/output

This scans the input folder for files like:
  LEN-10_HLA-B_07_02_ALG-MhcNuggetsI.scores.tsv.gz

and writes one HDF5 file per algorithm, e.g. MhcNuggetsI_percentiles.h5
Datasets are written at /HLA-B_07_02/10mer and contain a sorted 1D array
of scores (float32) using gzip compression.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd


FNAME_RE = re.compile(r"LEN-(?P<len>\d+)_HLA-(?P<allele>[A-Za-z0-9_]+)_ALG-(?P<alg>[^.]+)\.scores\.tsv(?:\.gz)?$")


def parse_filename(path: Path) -> Tuple[int, str, str]:
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    length = int(m.group("len"))
    allele = m.group("allele")
    alg = m.group("alg")
    return length, allele, alg


def find_score_column(df: pd.DataFrame) -> str:
    # case-insensitive search for 'score' in column names
    for col in df.columns:
        if "score" in col.lower():
            return col
    raise ValueError("No score-like column found in dataframe columns: " + ",".join(df.columns))


def process_file(path: Path) -> Tuple[str, int, str, np.ndarray]:
    length, allele, alg = parse_filename(path)
    df = pd.read_csv(path, sep="\t", compression=("gzip" if path.suffix == ".gz" else None))
    score_col = find_score_column(df)
    scores = df[score_col].to_numpy(dtype=np.float32)
    scores.sort()
    return alg, length, allele, scores


def group_files_by_algorithm(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        try:
            _, _, alg = parse_filename(p)
        except ValueError:
            continue
        groups.setdefault(alg, []).append(p)
    return groups


def build_hdf5_for_algorithm(algorithm: str, files: Iterable[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algorithm}_percentiles.h5"
    with h5py.File(out_path, "w") as h5:
        for p in sorted(files):
            alg, length, allele, scores = process_file(p)
            # group like /HLA-B_07_02/10mer
            grp_name = f"HLA-{allele}"
            ds_name = f"{length}mer"
            grp = h5.require_group(grp_name)
            # write dataset with gzip compression
            if ds_name in grp:
                print(f"Overwriting dataset {grp_name}/{ds_name} in {out_path}")
                del grp[ds_name]
            grp.create_dataset(ds_name, data=scores, compression="gzip", compression_opts=4)
            print(f"Stored {p.name} -> {out_path}:{grp_name}/{ds_name} (n={scores.size})")
    return out_path


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build HDF5 percentile files from TSV(.gz) score files")
    p.add_argument("input", type=Path, help="Input directory containing .tsv or .tsv.gz files for one or more algorithms")
    p.add_argument("--out-dir", type=Path, default=Path.cwd(), help="Directory to write HDF5 files")
    args = p.parse_args(argv)

    files = list(args.input.glob("*.scores.tsv")) + list(args.input.glob("*.scores.tsv.gz"))
    if not files:
        print(f"No .scores.tsv or .scores.tsv.gz files found in {args.input}")
        return 2

    groups = group_files_by_algorithm(files)
    if not groups:
        print("No files matched expected filename pattern; nothing to do.")
        return 3

    for alg, paths in groups.items():
        build_hdf5_for_algorithm(alg, paths, args.out_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
