#!/bin/bash

# Run all tests
echo "=== Running all tests ==="
python -m pytest

# If a specific test is provided, run only that test
if [ $# -eq 1 ]; then
    echo "=== Running specific test: $1 ==="
    python -m pytest tests/test_$1.py -v
fi