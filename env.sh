#!/bin/bash

# Define the path to the YAML file and the environment name
YAML_FILE="environment.yml"
ENV_NAME="llm-training-env"
# Check if the YAML file exists
if [ -f "$YAML_FILE" ]; then
    echo "YAML file found: $YAML_FILE"

    # Check if the environment already exists
    if conda env list | grep -q "$ENV_NAME"; then

        echo "Environment $ENV_NAME already exists. Deleting it..."
        conda env remove -n "$ENV_NAME"
    fi

    # Create the environment from the YAML file
    echo "Creating environment $ENV_NAME from $YAML_FILE..."
    conda env create -f "$YAML_FILE"
    conda activate "$ENV_NAME"
else
    echo "YAML file not found: $YAML_FILE"
    exit 1
fi