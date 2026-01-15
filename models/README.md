# Models Directory

This directory contains ONNX model files for stem separation. Models are excluded from git due to their size (~300MB+).

## Getting the htdemucs ONNX Model

The Demucs v4 (htdemucs) model needs to be converted from PyTorch to ONNX format. There are no pre-built downloads available - you must convert it yourself.

### Option 1: Automated Script (Recommended)

Run the provided script to automatically download and convert the model:

```bash
./scripts/download_model.sh
```

This handles all steps automatically and takes 5-10 minutes to complete.

### Option 2: Manual Conversion

If you prefer to convert manually or the script fails:

```bash
# 1. Clone with submodules (includes vendored dependencies)
git clone --recurse-submodules https://github.com/sevagh/demucs.onnx
cd demucs.onnx

# 2. Set up Python environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r scripts/requirements.txt

# 3. Convert PyTorch model to ONNX
python ./scripts/convert-pth-to-onnx.py ./onnx-models

# 4. Optimise ONNX model to ORT format (optional but recommended)
./scripts/convert-model-to-ort.sh

# 5. Copy model to stems project
# Use .ort (optimised) or .onnx (unoptimised)
cp ./onnx-models/htdemucs.ort /path/to/stems/models/htdemucs.onnx
# OR
cp ./onnx-models/htdemucs.onnx /path/to/stems/models/htdemucs.onnx
```

**Note**: The conversion process moves STFT/iSTFT outside the ONNX model (handled by our C++ code instead), which is why our implementation provides separate STFT preprocessing.

## Model Files

Place the following files in this directory:

- `htdemucs.onnx` or `htdemucs.ort` - Primary 4-stem model (vocals, drums, bass, other)
- `htdemucs_6s.onnx` (optional) - 6-stem model with piano and guitar (future support)

## Model Information

- **htdemucs**: ~300MB, 4 stems, 9.2 dB SDR quality
- **Input**: Stereo audio (any length, 44.1kHz recommended)
- **Output**: 4 stereo stems (vocals, drums, bass, other)
