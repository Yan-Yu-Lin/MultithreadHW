#!/usr/bin/env python3
"""
Multi-threaded 50×80 × 80×50 matrix multiply
Supports three “cases” from the assignment:

  --mode cell   → 50×50  = 2 500 threads (1 output cell each)
  --mode row    → 50×1   =     50 threads (1 row   each)
  --mode block  →  5×2   =     10 threads (10×25-cell blocks)

The heavy compute inside each thread is NumPy (which releases the GIL),
so you’ll observe real speed-ups even in CPython.
"""
import math, time, argparse, threading
import numpy as np

N_ROWS_A, N_COLS_A, N_COLS_B = 50, 80, 50   # dims as before

# ---------- matrix factory ---------- #
def build_matrices():
    i_idx = np.arange(N_ROWS_A).reshape(-1, 1)
    j_idx_A = np.arange(N_COLS_A)
    j_idx_B = np.arange(N_COLS_B)
    k_idx   = np.arange(N_COLS_A).reshape(-1, 1)

    A = 10.0 * np.sin(i_idx) - 0.6 * j_idx_A
    B = 7.0 - 12.0 * np.cos(5.0 * j_idx_B) + 1.8 * k_idx
    return A.astype(np.float64), B.astype(np.float64)

# ---------- worker strategies ---------- #
def indices_cell():
    """Iterator of (row_start,row_end,col_start,col_end) = 1 cell each."""
    for i in range(N_ROWS_A):
        for j in range(N_COLS_B):
            yield (i, i + 1, j, j + 1)

def indices_row():
    """Iterator of 1 full row of C at a time (row slices)."""
    for i in range(N_ROWS_A):
        yield (i, i + 1, 0, N_COLS_B)

def indices_block():
    """5×2 grid → 10 blocks of roughly 10 × 25 cells."""
    row_blocks = np.array_split(range(N_ROWS_A), 5)
    col_blocks = np.array_split(range(N_COLS_B), 2)
    for rs in row_blocks:
        for cs in col_blocks:
            yield (rs[0], rs[-1] + 1, cs[0], cs[-1] + 1)

def worker(A, B, C, rs, re, cs, ce):
    """Compute C[rs:re, cs:ce] without touching other regions."""
    C[rs:re, cs:ce] = A[rs:re, :] @ B[:, cs:ce]   # NumPy dot over k

def run_threaded(mode: str):
    A, B = build_matrices()
    C = np.zeros((N_ROWS_A, N_COLS_B), dtype=np.float64)

    strategy = {"cell": indices_cell,
                "row":  indices_row,
                "block":indices_block}[mode]

    threads = []
    for (rs, re, cs, ce) in strategy():
        t = threading.Thread(target=worker, args=(A, B, C, rs, re, cs, ce))
        threads.append(t); t.start()
    for t in threads: t.join()

    return C, len(threads)

# ---------- CLI ---------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cell", "row", "block"],
                        default="row",
                        help="partitioning scheme (default: row)")
    parser.add_argument("--runs", type=int, default=1,
                        help="number of repetitions (default 1)")
    args = parser.parse_args()

    for run in range(1, args.runs + 1):
        t0 = time.perf_counter()
        _, n_threads = run_threaded(args.mode)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        print(f"{args.mode},{n_threads},{run},{elapsed_ms:.3f}")

if __name__ == "__main__":
    main()
