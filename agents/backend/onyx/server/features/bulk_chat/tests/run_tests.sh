#!/bin/bash
# Script to run all tests

echo "Running bulk_chat tests..."
echo "=========================="

# Run all tests
pytest tests/ -v --cov=bulk_chat --cov-report=term-missing

# Run with coverage report
echo ""
echo "Generating coverage report..."
pytest tests/ --cov=bulk_chat --cov-report=html

echo ""
echo "Tests completed! Coverage report in htmlcov/"


