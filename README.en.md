# Matrix Multiplication Benchmark Report

This report summarizes the performance of different threading strategies for multiplying a 50×80 matrix by an 80×50 matrix using Python and NumPy, comparing it to a single-threaded baseline.

## Methodology

Four approaches were benchmarked:

1.  **Single-threaded (`single`):** Implemented in `matrix_mul_single.py`. Uses a standard triple-nested `for` loop (`for i`, `for j`, `for k`) to calculate each element of the result matrix sequentially.
2.  **Multi-threaded (`cell`, `row`, `block`):** Implemented in `matrix_mul_threaded.py`. This version partitions the work and uses multiple threads.

    *   **Partitioning:** Instead of a single triple loop, the work is divided based on the chosen mode:
        *   `cell`: One thread per output cell.
        *   `row`: One thread per output row.
        *   `block`: One thread per 10x25 block of the output matrix.
    *   **Threading:** The `run_threaded` function creates `threading.Thread` objects, assigning each a portion of the calculation via the `worker` function. The `worker` function uses NumPy's optimized matrix multiplication (`@`) for its assigned sub-matrix.
    *   **Synchronization:** The main thread waits for all worker threads to complete using `t.join()` before proceeding.

The benchmarks were run 3 times for each approach using Python 3.13.2 within a `uv` virtual environment on macOS.

**Execution Time Measurement (Q2):** The CPU running time was measured in milliseconds using Python's `time.perf_counter()`. The time was recorded just before the multiplication function call (`multiply_single` or `run_threaded`) and again immediately after. The difference between these two timestamps was then multiplied by 1,000 to convert seconds to milliseconds (`elapsed_ms = (time.perf_counter() - t0) * 1_000`).

## Results (Q2)

The average elapsed time (in milliseconds) for each mode over 3 runs is presented below:

| Mode          | Threads | Average Time (ms) | Speedup vs. Single |
| :------------ | ------: | ----------------: | :----------------- |
| `single`      |       1 |             29.28 | 1.0x               |
| `cell`        |    2500 |            118.58 | 0.25x (Slower)     |
| `row`         |      50 |              2.29 | 12.8x              |
| `block`       |      10 |              1.14 | 25.7x              |

*(Raw data in `results.csv`)*

## Discoveries and Comments (Q3)

This exercise highlights several key aspects of threaded programming:

-   **Overhead vs. Parallelism:** The most significant discovery is the trade-off between the potential for parallelism and the overhead introduced by managing threads. The `cell` mode, despite having the most threads (2500), performed significantly worse than the single-threaded version. This demonstrates that creating and coordinating a very large number of threads for extremely small tasks (calculating one cell) incurs more overhead than the computational benefit gained.

-   **Task Granularity Matters:** Performance improved dramatically when threads were given larger, more substantial tasks. The `row` mode (50 threads, 1 row each) was ~13x faster than single-threaded, and the `block` mode (10 threads, 1 block each) was ~26x faster. This shows the importance of choosing an appropriate granularity for parallel tasks.

-   **Leveraging Optimized Libraries (NumPy & GIL):** The speedups observed in `row` and `block` modes are possible even with Python's Global Interpreter Lock (GIL) because the core computation within each thread (`A[rs:re, :] @ B[:, cs:ce]`) is performed by NumPy. NumPy releases the GIL during its C-optimized operations, allowing true parallel execution of the matrix multiplication across multiple CPU cores.

-   **Optimal Strategy:** For this specific problem (50x80 @ 80x50 matrix multiplication) on the test system, the `block` strategy with 10 threads provided the best balance, minimizing overhead while maximizing parallel computation through NumPy.

## Conclusion

Multi-threading can significantly accelerate computationally intensive tasks like matrix multiplication, but careful consideration must be given to how the work is partitioned. Dividing the work into appropriately sized chunks (task granularity) is crucial to avoid the performance penalties associated with excessive thread management overhead. Utilizing libraries like NumPy that release the GIL allows Python threads to achieve substantial performance gains on multi-core processors.
