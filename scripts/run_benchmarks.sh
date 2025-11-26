#!/bin/bash
# Run performance benchmarks for Quantum Conduit
#
# Usage: ./scripts/run_benchmarks.sh [benchmark_name]
#
# If no argument provided, runs all benchmarks.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARKS_DIR="$PROJECT_ROOT/benchmarks"
REPORTS_DIR="$PROJECT_ROOT/benchmarks/reports"

# Create reports directory
mkdir -p "$REPORTS_DIR"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Function to run a benchmark
run_benchmark() {
    local bench_name=$1
    local bench_file="$BENCHMARKS_DIR/$bench_name.py"
    
    if [ ! -f "$bench_file" ]; then
        echo "Error: Benchmark file not found: $bench_file"
        return 1
    fi
    
    echo "Running $bench_name..."
    python "$bench_file" 2>&1 | tee "$REPORTS_DIR/${bench_name}.txt"
    echo "Results saved to $REPORTS_DIR/${bench_name}.txt"
    echo ""
}

# Main execution
if [ $# -eq 0 ]; then
    # Run all benchmarks
    echo "Running all benchmarks..."
    echo "================================"
    echo ""
    
    for bench_file in "$BENCHMARKS_DIR"/bench_*.py; do
        if [ -f "$bench_file" ]; then
            bench_name=$(basename "$bench_file" .py)
            run_benchmark "$bench_name"
        fi
    done
    
    echo "All benchmarks completed!"
    echo "Results saved to $REPORTS_DIR/"
else
    # Run specific benchmark
    bench_name=$1
    if [[ ! "$bench_name" =~ ^bench_ ]]; then
        bench_name="bench_${bench_name}"
    fi
    run_benchmark "$bench_name"
fi

