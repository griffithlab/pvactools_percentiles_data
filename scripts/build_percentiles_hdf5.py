#!/usr/bin/env python3
"""Build HDF5 percentile reference files from .tsv(.gz) score files.

Each input file is expected to follow the filename pattern:
    LEN-<length>_HLA-<allele>_ALG-<algorithm>.scores.tsv[.gz]

The script scans an input directory for all matching .scores.tsv / .scores.tsv.gz
files and writes one HDF5 file per allele. Inside each allele HDF5 file the
stored datasets are grouped by algorithm name, and within each algorithm group
there is one dataset per peptide length named like "10mer" or "9mer". Each
dataset contains a sorted 1D numpy array (dtype float32) of scores saved with
gzip compression. These sorted arrays can be used to compute percentiles via
binary search or numpy/searchsorted.

Usage: python build_percentiles_hdf5.py /path/to/input_folder --out-dir /path/to/output

Example filename that will be parsed correctly:
    LEN-10_HLA-B_07_02_ALG-MhcNuggetsI.scores.tsv.gz

Example on-disk layout produced by the script (one file per allele):
    HLA-B_07_02_percentiles.h5
    └─ /MhcNuggetsI/10mer         (1D float32 array, sorted)
    └─ /NetMHCpan/9mer             (1D float32 array, sorted)

Small example showing how to open an allele HDF5 file and access the stored
scores for a particular algorithm and peptide length in Python::

        import h5py
        import numpy as np

        h5_path = "HLA-B_07_02_percentiles.h5"
        alg = "MhcNuggetsI"
        length = 10

        with h5py.File(h5_path, "r") as h5:
                ds_path = f"/{alg}/{length}mer"
                if ds_path in h5:
                        scores = h5[ds_path][()]  # numpy array of float32, already sorted
                else:
                        raise KeyError(f"Dataset not found: {ds_path} in {h5_path}")

        # Example: compute the percentile of a new score using searchsorted
        new_score = 0.123
        pct = 100.0 * np.searchsorted(scores, new_score, side="left") / scores.size

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


def group_files_by_allele(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        try:
            _, allele, _ = parse_filename(p)
        except ValueError:
            continue
        groups.setdefault(allele, []).append(p)
    return groups


def build_hdf5_for_allele(allele: str, files: Iterable[Path], out_dir: Path) -> Path:
    """Create one HDF5 file per allele containing datasets at /<Algorithm>/<Length>mer.

    Args:
        allele: allele string (e.g. 'A_01_01')
        files: Iterable of Path objects for this allele
        out_dir: directory to write the HDF5 file

    Returns:
        Path to the written HDF5 file
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"HLA-{allele}_percentiles.h5"
    with h5py.File(out_path, "w") as h5:
        # For each file, process and store under /<Algorithm>/<Length>mer
        for p in sorted(files):
            alg, length, file_allele, scores = process_file(p)
            # sanity: file_allele should match allele
            if file_allele != allele:
                # this shouldn't happen since files were grouped by allele, but skip if it does
                print(f"Skipping {p.name}: allele mismatch ({file_allele} != {allele})")
                continue
            grp = h5.require_group(alg)
            ds_name = f"{length}mer"
            # overwrite if present
            if ds_name in grp:
                print(f"Overwriting dataset /{alg}/{ds_name} in {out_path}")
                del grp[ds_name]
            grp.create_dataset(ds_name, data=scores, compression="gzip", compression_opts=4)
            print(f"Stored {p.name} -> {out_path}:/{alg}/{ds_name} (n={scores.size})")
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

    allele_groups = group_files_by_allele(files)
    if not allele_groups:
        print("No files matched expected filename pattern; nothing to do.")
        return 3

    for allele, paths in allele_groups.items():
        build_hdf5_for_allele(allele, paths, args.out_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
