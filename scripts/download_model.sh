#!/bin/bash
# Download and convert Demucs htdemucs model to ONNX format
# This script automates the process described in models/README.md

set -euo pipefail

# Configuration
DEMUCS_ONNX_REPO="https://github.com/sevagh/demucs.onnx"
WORK_DIR="${TMPDIR:-/tmp}/demucs_conversion"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_OUTPUT="${PROJECT_ROOT}/models/htdemucs.onnx"

echo "=== Demucs Model Conversion ==="
echo "This will download and convert the Demucs model to ONNX format"
echo "Expected time: 5-10 minutes"
echo "Expected size: ~300MB"
echo ""

# Check for required tools
command -v git >/dev/null 2>&1 || { echo "Error: git is required"; exit 1; }

# Find suitable Python version (3.9-3.12, onnxruntime doesn't support 3.13+ yet)
PYTHON_CMD=""
for pyver in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$pyver" >/dev/null 2>&1; then
        PY_VERSION=$("$pyver" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 9 ] && [ "$PY_MINOR" -le 12 ]; then
            PYTHON_CMD="$pyver"
            echo "Found suitable Python: $pyver (version $PY_VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.9-3.12 required (onnxruntime not yet compatible with 3.13+)"
    echo "Install with: brew install python@3.12"
    exit 1
fi

# Create work directory
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "Step 1/5: Cloning demucs.onnx repository..."
git clone --recurse-submodules --depth 1 "${DEMUCS_ONNX_REPO}"
cd demucs.onnx

echo ""
echo "Step 2/5: Setting up Python environment..."
"$PYTHON_CMD" -m venv venv
source venv/bin/activate

echo ""
echo "Step 3/5: Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r scripts/requirements.txt

echo ""
echo "Step 4/5: Converting PyTorch model to ONNX (this may take several minutes)..."
python ./scripts/convert-pth-to-onnx.py ./onnx-models

echo ""
echo "Step 5/5: Optimising to ORT format..."
if ./scripts/convert-model-to-ort.sh 2>/dev/null; then
    # Use optimised ORT format if available
    MODEL_SOURCE="./onnx-models/htdemucs.ort"
    echo "✓ Created optimised ORT model"
else
    # Fall back to standard ONNX format
    MODEL_SOURCE="./onnx-models/htdemucs.onnx"
    echo "⚠ ORT optimisation failed, using standard ONNX format"
fi

# Verify model was created
if [ ! -f "${MODEL_SOURCE}" ]; then
    echo ""
    echo "Error: Model file not found at ${MODEL_SOURCE}"
    echo "Conversion may have failed"
    exit 1
fi

# Check file size
MODEL_SIZE=$(stat -f%z "${MODEL_SOURCE}" 2>/dev/null || stat -c%s "${MODEL_SOURCE}")
MODEL_SIZE_MB=$((MODEL_SIZE / 1024 / 1024))

if [ "${MODEL_SIZE_MB}" -lt 100 ]; then
    echo ""
    echo "Error: Model file is too small (${MODEL_SIZE_MB}MB)"
    echo "Expected approximately 300MB"
    exit 1
fi

echo ""
echo "Copying model to project..."
mkdir -p "$(dirname "${MODEL_OUTPUT}")"
cp "${MODEL_SOURCE}" "${MODEL_OUTPUT}"

echo ""
echo "=== Conversion Complete ==="
echo "Model saved to: ${MODEL_OUTPUT}"
echo "Size: ${MODEL_SIZE_MB}MB"
echo ""
echo "You can now run: ./stems your_audio.wav"
echo ""
echo "Cleaning up temporary files..."
cd /
rm -rf "${WORK_DIR}"

echo "✓ Done!"
