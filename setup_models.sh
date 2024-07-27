#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

MODEL_DIR="${1:-$SCRIPT_DIR/models}"
mkdir -p "$MODEL_DIR/piper"
mkdir -p "$MODEL_DIR/whisper"

# Download piper voice model and config
# See https://github.com/rhasspy/piper/blob/master/VOICES.md for voice options
wget 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true' -O "$MODEL_DIR/piper/voice.onnx"
wget 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true.json' -O "$MODEL_DIR/piper/voice.json"

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Cleanup function to remove the temporary directory on script exit
cleanup() {
    echo "Cleaning up temporary directory: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
}

# Set the cleanup function to run on script exit
trap cleanup EXIT

# Follow steps in https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md to setup the model
# See https://github.com/ml-explore/mlx-examples/blob/main/whisper/convert.py for model options
git clone https://github.com/ml-explore/mlx-examples.git "$TEMP_DIR/mlx-examples"
cd "$TEMP_DIR/mlx-examples/whisper"
python convert.py --torch-name-or-path large-v3 -q --q_bits 4 --mlx-path $MODEL_DIR/whisper
cd -
