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