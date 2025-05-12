#!/usr/bin/env python3
"""
Single-threaded 50×80 × 80×50 matrix multiply
(for OS HW #2 baseline measurement)
"""
import math, time, argparse
import numpy as np

N_ROWS_A = 50   # rows of A, rows of C
N_COLS_A = 80   # cols of A, rows of B
N_COLS_B = 50   # cols of B, cols of C

def build_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Populate A (50 × 80) and B (80 × 50) using the hand-out formulas."""
    i_idx = np.arange(N_ROWS_A).reshape(-1, 1)              # column vector (i)
    j_idx_A = np.arange(N_COLS_A)                           # row   vector (j)
    j_idx_B = np.arange(N_COLS_B)                           # row   vector (j)
    k_idx = np.arange(N_COLS_A).reshape(-1, 1)              # for B’s rows

    A = 10.0 * np.sin(i_idx) - 0.6 * j_idx_A               # broadcast over i, j
    B = 7.0 - 12.0 * np.cos(5.0 * j_idx_B) + 1.8 * k_idx   # broadcast over k, j
    return A.astype(np.float64), B.astype(np.float64)

def multiply_single(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Classic triple-for loop (so we *feel* the threading benefit later)."""
    C = np.zeros((N_ROWS_A, N_COLS_B), dtype=np.float64)
    for i in range(N_ROWS_A):
        for j in range(N_COLS_B):
            s = 0.0
            for k in range(N_COLS_A):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1,
                        help="number of repetitions (default 1)")
    args = parser.parse_args()

    A, B = build_matrices()
    for run in range(1, args.runs + 1):
        t0 = time.perf_counter()
        _ = multiply_single(A, B)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        print(f"single,1,{run},{elapsed_ms:.3f}")

if __name__ == "__main__":
    main()
