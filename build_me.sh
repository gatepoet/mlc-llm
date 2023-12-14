#!/bin/bash

# Function to execute the MLC LLM build command
execute_command() {
    local action=$1
    local quantization="q4f16_1"

    python -m mlc_llm.build --model ./dist/models/$MODEL \
                            --no-cutlass-attn --no-cutlass-norm \
                            --use-cuda-graph --num-shards $GPUS \
                            $action --quantization $quantization \
                            --use-presharded-weights --target cuda
}

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 model gpus [action]"
    exit 1
fi

# Assigning input arguments to variables
MODEL=$1
GPUS=$2
ACTION=$3

# Default actions
CONVERT_ACTION="--convert-weights-only"
BUILD_ACTION="--build-model-only"

# Execute commands based on the provided action
if [ -z "$ACTION" ]; then
    # If no action is specified, run both actions
    execute_command $CONVERT_ACTION
    execute_command $BUILD_ACTION
elif [ "$ACTION" = "weights" ]; then
    # If action is weights, run convert weights only
    execute_command $CONVERT_ACTION
elif [ "$ACTION" = "build" ]; then
    # If action is build, run build model only
    execute_command $BUILD_ACTION
else
    echo "Invalid action: $ACTION. Valid actions are 'weights' or 'build'."
    exit 1
fi
