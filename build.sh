#!/bin/bash
SOURCE=${1:-hspeedtrack.cc}
OUTPUT=${2:-hspeedtrack}
g++ -g -std=c++20 -fopenmp -O5 -march=native -o "$OUTPUT" "$SOURCE" `pkg-config --libs opencv4` \
    -isystem /usr/include/opencv4 \
    -isystem /usr/local/cuda/include \
    -isystem /usr/include/x86_64-linux-gnu \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/x86_64-linux-gnu \
    -lopencv_highgui -lnvinfer -lnvonnxparser -lcudart -pthread
