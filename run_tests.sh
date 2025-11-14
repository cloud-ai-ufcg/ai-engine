#!/bin/bash

# Test runner script for AI Engine no-fallback tests
# This script sets up the environment and runs the unit tests

set -e

echo "🧪 AI Engine No-Fallback Tests"
echo "================================"

# Change to ai-engine directory 
cd "$(dirname "$0")"

echo "📦 Installing test dependencies..."
# Install pytest if not already installed
pip install pytest pytest-mock

echo "🔍 Running unit tests..."
python -m pytest test_agents_no_fallback.py -v --tb=short

echo "✅ Tests completed!"