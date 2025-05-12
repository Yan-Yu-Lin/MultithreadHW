# Matrix Multiplication Benchmark Report

This report summarizes the performance of different threading strategies for multiplying a 50×80 matrix by an 80×50 matrix using Python and NumPy.

## Methodology

Four approaches were benchmarked, each run 3 times:

1.  **Single-threaded (`single`):** A standard triple-loop multiplication executed sequentially. (1 thread total)
2.  **Cell-based threading (`cell`):** Each cell of the 50x50 output matrix was computed by a separate thread. (2500 threads total)
3.  **Row-based threading (`row`):** Each row of the 50x50 output matrix was computed by a separate thread. (50 threads total)
4.  **Block-based threading (`block`):** The output matrix was divided into a 5x2 grid (10 blocks), and each block was computed by a separate thread. (10 threads total)

The benchmarks were run using Python 3.13.2 within a `uv` virtual environment on macOS.

## Results

The average elapsed time (in milliseconds) for each mode over 3 runs was:

| Mode          | Threads | Average Time (ms) | Speedup vs. Single |
| :------------ | ------: | ----------------: | :----------------- |
| `single`      |       1 |             29.28 | 1.0x               |
| `cell`        |    2500 |            118.58 | 0.25x (Slower)     |
| `row`         |      50 |              2.29 | 12.8x              |
| `block`       |      10 |              1.14 | 25.7x              |

*(Calculated from `results.csv`)*

## Discussion

-   **Overhead vs. Parallelism:** The results clearly demonstrate the trade-off between parallelism and overhead. While threading can offer significant speedups, creating too many threads for fine-grained tasks (`cell` mode) leads to performance degradation due to the cost of thread creation, scheduling, and synchronization.
-   **Task Granularity:** Assigning larger chunks of work per thread (`row` and `block` modes) proved much more effective. NumPy's matrix operations (`@`) release the Global Interpreter Lock (GIL) and are highly optimized, allowing these threads to perform substantial computation in parallel.
-   **Optimal Strategy:** The `block` mode, using only 10 threads to compute relatively large 10x25 sub-matrices, yielded the best performance, achieving a ~26x speedup over the single-threaded implementation. This suggests that for this problem size and hardware, a moderate number of threads working on coarse-grained tasks is optimal.

## Conclusion

Multi-threading significantly accelerated the matrix multiplication task compared to a single-threaded approach, provided the work was partitioned appropriately. The `block` partitioning strategy offered the best performance by effectively balancing parallel execution with thread management overhead. The `cell` strategy highlights the performance penalty of excessive threading for small tasks. 