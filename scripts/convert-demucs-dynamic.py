#!/usr/bin/env python3
"""
Modified Demucs to ONNX converter with dynamic input shapes.
Based on sevagh/demucs.onnx conversion script but adds dynamic_axes support.
"""

import sys
import torch
from torch.nn import functional as F
from pathlib import Path

# Add vendor/demucs to path if running from demucs.onnx repo
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "demucs"))

from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs, standalone_spec, standalone_magnitude

def convert_demucs_with_dynamic_shapes(output_dir: Path, model_name: str = "htdemucs"):
    """Convert Demucs model to ONNX with dynamic input dimensions."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {model_name} model...")
    model = get_model(model_name)

    # Extract core HTDemucs model
    if isinstance(model, HTDemucs):
        core_model = model
    elif hasattr(model, 'models') and isinstance(model.models[0], HTDemucs):
        core_model = model.models[0]
    else:
        raise TypeError("Unsupported model type")

    core_model.eval()

    # Prepare dummy input (size doesn't matter since we use dynamic axes)
    # Use segment length as a reasonable default
    training_length = int(core_model.segment * core_model.samplerate)
    dummy_waveform = torch.randn(1, 2, training_length)

    # Pre-pad if needed
    if dummy_waveform.shape[-1] < training_length:
        dummy_waveform = F.pad(dummy_waveform, (0, training_length - dummy_waveform.shape[-1]))

    # Compute spectrogram for the dummy input
    magspec = standalone_magnitude(standalone_spec(dummy_waveform))

    dummy_input = (dummy_waveform, magspec)

    # Define output path
    onnx_file = output_dir / f"{model_name}.onnx"

    print(f"Exporting to ONNX with dynamic shapes...")
    print(f"  Waveform shape: {dummy_waveform.shape}")
    print(f"  Spectrogram shape: {magspec.shape}")

    # Export with dynamic axes for variable-length audio
    try:
        torch.onnx.export(
            core_model,
            dummy_input,
            onnx_file,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input', 'x'],
            output_names=['output', 'add_67'],
            dynamic_axes={
                'input': {2: 'time'},       # Time dimension is dynamic
                'x': {3: 'time_freq'},      # Frequency time dimension is dynamic
                'output': {4: 'time_freq'}, # Output time dimension
                'add_67': {3: 'time'}       # Output waveform time dimension
            }
        )
        print(f"✓ Model successfully exported to {onnx_file}")
        print(f"  File size: {onnx_file.stat().st_size / 1024 / 1024:.1f} MB")

        # Check for external data file
        data_file = Path(str(onnx_file) + ".data")
        if data_file.exists():
            print(f"  External data: {data_file.stat().st_size / 1024 / 1024:.1f} MB")

        return True
    except Exception as e:
        print(f"✗ Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert Demucs PyTorch model to ONNX with dynamic input shapes'
    )
    parser.add_argument(
        'dest_dir',
        type=Path,
        help='destination directory for the converted model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='htdemucs',
        help='model name (default: htdemucs)'
    )

    args = parser.parse_args()

    success = convert_demucs_with_dynamic_shapes(args.dest_dir, args.model)
    sys.exit(0 if success else 1)
