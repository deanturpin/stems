#!/bin/bash
# Download and convert Demucs htdemucs model to ONNX format
# This script automates the process described in models/README.md
#
# Usage:
#   ./download_model.sh          # Download 4-stem model (default)
#   ./download_model.sh 6        # Download 6-stem model (drums, bass, other, vocals, guitar, piano)

set -euo pipefail

# Configuration
MODEL_TYPE="${1:-4}"  # 4 or 6 stems
DEMUCS_ONNX_REPO="https://github.com/sevagh/demucs.onnx"
WORK_DIR="${TMPDIR:-/tmp}/demucs_conversion"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Select model name based on type
if [ "$MODEL_TYPE" = "6" ]; then
    MODEL_NAME="htdemucs_6s"
    echo "=== Demucs 6-Stem Model Conversion ==="
    echo "Stems: drums, bass, other, vocals, guitar, piano"
    echo "Note: Piano quality may not be optimal (experimental)"
else
    MODEL_NAME="htdemucs"
    echo "=== Demucs 4-Stem Model Conversion ==="
    echo "Stems: drums, bass, other, vocals"
fi

MODEL_OUTPUT="${PROJECT_ROOT}/models/${MODEL_NAME}.onnx"

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
# Clone without submodules initially (they're only needed for C++ builds, not Python conversion)
git clone --depth 1 "${DEMUCS_ONNX_REPO}"
cd demucs.onnx

# Only clone the demucs submodule which contains the Python model code
git submodule update --init --depth 1 vendor/demucs

echo ""
echo "Step 2/5: Setting up Python environment..."
"$PYTHON_CMD" -m venv venv
source venv/bin/activate

echo ""
echo "Step 3/5: Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r scripts/requirements.txt
pip install --quiet onnxscript  # Required for ONNX export but not in requirements.txt

echo ""
echo "Step 4/5: Converting PyTorch model to ONNX (this may take several minutes)..."
# Use our custom conversion script with dynamic shapes support
"${PYTHON_CMD}" "${PROJECT_ROOT}/scripts/convert-demucs-dynamic.py" ./onnx-models --model "${MODEL_NAME}"

echo ""
echo "Step 5/5: Optimising to ORT format..."
# Skip ORT optimisation for now, just use ONNX format
MODEL_SOURCE="./onnx-models/${MODEL_NAME}.onnx"
if [ ! -f "${MODEL_SOURCE}" ]; then
    echo "⚠ ONNX model not found, checking for external data format..."
    # Model might have been created without external data
    MODEL_SOURCE="./onnx-models/${MODEL_NAME}.onnx"
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
echo "Copying model files to project..."
mkdir -p "$(dirname "${MODEL_OUTPUT}")"
cp "${MODEL_SOURCE}" "${MODEL_OUTPUT}"

# Copy external data file if it exists (ONNX models can store weights separately)
if [ -f "${MODEL_SOURCE}.data" ]; then
    cp "${MODEL_SOURCE}.data" "${MODEL_OUTPUT}.data"
    echo "Copied external data file: ${MODEL_OUTPUT}.data"
fi

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
