#!/bin/bash

# APEGA Ingestion Test Script
echo "🚀 Running APEGA Ingestion Test..."

# Make sure we're in the right directory
cd /Users/logan/Git/Agents/APEGA

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Run diagnostics first
echo "🔧 Running diagnostics..."
python diagnose.py

echo ""
echo "📄 Running document ingestion..."
python ingest.py --directory Context

echo ""
echo "✅ Ingestion test complete!"