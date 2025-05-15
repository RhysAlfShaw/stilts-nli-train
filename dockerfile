FROM continuumio/miniconda3

# Install build tools required for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc
ENV CXX=g++

# Copy environment file and create conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the conda environment in following RUN commands
SHELL ["conda", "run", "-n", "llm-training-env", "/bin/bash", "-c"]

# Install llama-cpp-python (CPU build)
RUN pip install llama-cpp-python

# Set environment variables
ENV LLAMA_CPP_LOG_LEVEL=INFO

WORKDIR /workspace
CMD ["conda", "run", "-n", "llama", "python"]
