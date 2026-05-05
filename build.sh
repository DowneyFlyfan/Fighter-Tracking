#!/bin/bash
SOURCE=${1:-main.cc}
OUTPUT=${2:-main}

# Detect CPU architecture (uname -m: machine hardware name)
ARCH=$(uname -m)

# Multiarch triplet for system header/lib lookup.
# x86_64 (Intel/AMD 64-bit) -> x86_64-linux-gnu
# aarch64 (ARM 64-bit, e.g. Jetson Orin Nano) -> aarch64-linux-gnu
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    TRIPLET="aarch64-linux-gnu"
    ARCH_FLAGS="-mcpu=native"
else
    TRIPLET="x86_64-linux-gnu"
    ARCH_FLAGS="-march=native"
fi

# Prefer OpenCV 5 (built into /usr/local) over apt OpenCV 4 if available.
if [ -f /usr/local/lib/pkgconfig/opencv5.pc ]; then
    OCV_PKG=opencv5
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
else
    OCV_PKG=opencv4
fi

# CPU build (no NN runtime; pure C++ + OpenCV).
g++ -g -std=c++20 -O3 $ARCH_FLAGS -o "$OUTPUT" "$SOURCE" \
    `pkg-config --cflags --libs $OCV_PKG` \
    -isystem /usr/include/$TRIPLET \
    -L/usr/lib/$TRIPLET \
    -lopencv_highgui -pthread
