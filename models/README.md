# Models Directory

This directory contains ONNX model files for stem separation. Models are excluded from git due to their size (~300MB+).

## Getting the htdemucs ONNX Model

The Demucs v4 (htdemucs) model needs to be converted from PyTorch to ONNX format.

### Option 1: Use sevagh/demucs.onnx (Recommended)

Clone and convert from the reference implementation:

```bash
git clone https://github.com/sevagh/demucs.onnx
cd demucs.onnx
python ./scripts/convert-pth-to-onnx.py ./demucs-onnx
cp ./demucs-onnx/htdemucs.ort ../models/
```

### Option 2: Convert from official Demucs

```bash
# Install dependencies
pip install demucs onnx

# Download and convert model
# (Instructions to be added once conversion script is ready)
```

## Model Files

Place the following files in this directory:

- `htdemucs.onnx` or `htdemucs.ort` - Primary 4-stem model (vocals, drums, bass, other)
- `htdemucs_6s.onnx` (optional) - 6-stem model with piano and guitar (future support)

## Model Information

- **htdemucs**: ~300MB, 4 stems, 9.2 dB SDR quality
- **Input**: Stereo audio (any length, 44.1kHz recommended)
- **Output**: 4 stereo stems (vocals, drums, bass, other)
