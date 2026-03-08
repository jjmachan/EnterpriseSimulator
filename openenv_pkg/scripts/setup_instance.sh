#!/bin/bash
# Setup script for H100 instance (Northflank)
# Run this after uploading the project to /workspace/

set -e

echo "=== H100 Instance Setup ==="

# 1. Install system deps
echo "Installing system packages..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv curl git sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Create venv and install deps
echo "Setting up Python environment..."
cd /workspace/openenv_pkg

uv venv --python 3.11
source .venv/bin/activate

# Install the openenv package
uv pip install -e .

# Install ML deps: vLLM for inference, datasets for formatting
uv pip install vllm datasets

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Set OPENAI_API_KEY:  export OPENAI_API_KEY='sk-...'"
echo "  2. Start vLLM:          python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-4B --port 8001 --dtype bfloat16 --gpu-memory-utilization 0.5 &"
echo "  3. Wait for vLLM to load, then run data collection:"
echo "     cd /workspace/openenv_pkg && python scripts/collect_data.py"
echo ""
