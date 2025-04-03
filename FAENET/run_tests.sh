#!/bin/bash
# Script to run all tests for the FAENET package

# Change to the project root directory
cd "$(dirname "$0")"

# Run the tests using pytest
python -m pytest tests -v

# Check if any tests failed
if [ $? -eq 0 ]; then
    echo -e "\n\033[1;32mAll tests passed!\033[0m"
else
    echo -e "\n\033[1;31mSome tests failed!\033[0m"
    exit 1
fi