#!/usr/bin/env fish
# bench.fish – run the matrix‑multiply benchmarks under fish + uv

set -l VENV ".venv"

# -----------------------------------------------------------
# 1. Ensure the uv virtual‑env exists and is activated
# -----------------------------------------------------------
if not test -d $VENV
    echo "Creating virtual environment with uv …"
    uv venv $VENV
end

# If the venv isn't already on PATH, source its activation script
if not string match -q "$PWD/$VENV/bin"* (string split ":" $PATH)
    source $VENV/bin/activate.fish
end

# -----------------------------------------------------------
# 2. Guarantee NumPy is importable inside this interpreter
# -----------------------------------------------------------
# Use the specific python from the venv
$VENV/bin/python -c "import numpy" >/dev/null 2>&1
set numpy_missing $status   # 0 = present, non‑zero = missing

if test $numpy_missing -ne 0
    echo "Installing NumPy into the venv …"
    # This should install into the activated venv implicitly,
    # but let's ensure it finishes before proceeding.
    uv pip install --quiet numpy
    # Re-check just to be sure, though uv install should be synchronous
    $VENV/bin/python -c "import numpy" >/dev/null 2>&1
    if test $status -ne 0
        echo "Error: Failed to install or find NumPy even after attempting installation." >&2
        exit 1
    end
end

# -----------------------------------------------------------
# 3. Run the benchmarks
# -----------------------------------------------------------
set RESULTS results.csv
echo "mode,threads,run,elapsed_ms" > $RESULTS

# Use the specific python from the venv
# Single‑thread baseline (3 runs)
$VENV/bin/python matrix_mul_single.py --runs 3 >> $RESULTS

# Multithread cases (3 runs each)
for i in (seq 3)
    $VENV/bin/python matrix_mul_threaded.py --mode cell  >> $RESULTS
    $VENV/bin/python matrix_mul_threaded.py --mode row   >> $RESULTS
    $VENV/bin/python matrix_mul_threaded.py --mode block >> $RESULTS
end

echo "Benchmark complete. Results saved to $RESULTS"

# Display the results in a formatted table
echo ""
echo "Summary Table:"
column -t -s "," $RESULTS
