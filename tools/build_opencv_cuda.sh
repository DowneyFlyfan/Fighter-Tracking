#!/usr/bin/env bash
# Build & install OpenCV with full CUDA support (incl. cudacodec / NVDEC)
# for an NVIDIA RTX 5070 Ti (Blackwell, sm_120) on Ubuntu + CUDA 13.2.
#
# Usage (one-shot, ~30-50 min on a modern 16-core CPU):
#     sudo bash tools/build_opencv_cuda.sh
#
# Result:
#     /usr/local/lib/libopencv_*.so          (incl. libopencv_cudacodec.so)
#     /usr/local/lib/python3.*/site-packages/cv2/  (Python bindings)
#     /usr/local/include/opencv4/opencv2/cudacodec.hpp
#     /usr/local/lib/pkgconfig/opencv4.pc    (overrides Ubuntu's apt OpenCV)
#
# After install, your existing build.sh / Python imports keep working but now
# use the CUDA-enabled OpenCV. Verify with:
#     python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
#     python3 -c "import cv2; print(hasattr(cv2, 'cudacodec'))"

set -euo pipefail

# ---------- Tunables ----------
OPENCV_BRANCH=5.x                  # main development branch; has CUDA 13 fixes
INSTALL_PREFIX=/usr/local          # change to /opt/opencv-cuda if you prefer
SRC_ROOT=/opt/opencv_build         # where sources live; user-owned after this
CUDA_ARCH="12.0"                   # RTX 5070 Ti (Blackwell) compute capability
JOBS=$(nproc)

# Optional: cudacodec needs the *real* NVIDIA Video Codec SDK headers
# (nvcuvid.h / cuviddec.h / nvEncodeAPI.h with direct API, not the
# dlsym-based ones in FFmpeg's nv-codec-headers). If you have the SDK,
# point VCSDK_INCLUDE at its Interface/ directory; otherwise cudacodec
# will be skipped and the rest of the CUDA modules will still build.
#   Get it from: https://developer.nvidia.com/video-codec-sdk
#   Then: export VCSDK_INCLUDE=/path/to/Video_Codec_SDK_*/Interface
VCSDK_INCLUDE=${VCSDK_INCLUDE:-}

# Detect the user who invoked sudo (so the source tree isn't root-owned)
SUDO_USER_NAME=${SUDO_USER:-$USER}

echo "============================================================"
echo "OpenCV $OPENCV_BRANCH (5.x dev branch) + CUDA build"
echo "  prefix      : $INSTALL_PREFIX"
echo "  source      : $SRC_ROOT"
echo "  arch        : sm_$CUDA_ARCH (RTX 5070 Ti / Blackwell)"
echo "  user        : $SUDO_USER_NAME"
echo "  jobs        : $JOBS"
echo "  cudacodec   : $([ -n "$VCSDK_INCLUDE" ] && echo "ON (SDK at $VCSDK_INCLUDE)" || echo "OFF (set VCSDK_INCLUDE to enable)")"
echo "============================================================"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: must run as root (sudo bash $0)"
    exit 1
fi

# ---------- 1. System deps ----------
# Check first — only invoke apt if something is actually missing. This avoids
# triggering unrelated dpkg post-install failures (e.g., a broken
# nvidia-dkms-* package that has nothing to do with our build).
echo
echo "[1/6] Checking build deps..."
DEPS=(build-essential cmake ninja-build git pkg-config
      libavcodec-dev libavformat-dev libswscale-dev libavutil-dev
      libtbb-dev libgtk-3-dev
      libjpeg-dev libpng-dev libtiff-dev libwebp-dev
      libv4l-dev libxvidcore-dev libx264-dev
      python3-dev python3-numpy)
MISSING=()
for p in "${DEPS[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed"; then
        MISSING+=("$p")
    fi
done
if [ ${#MISSING[@]} -eq 0 ]; then
    echo "  all build deps already present, skipping apt"
else
    echo "  installing missing: ${MISSING[*]}"
    # Use --no-install-recommends and tolerate unrelated dpkg failures (e.g.
    # broken nvidia-dkms): only abort if our explicit packages still missing.
    apt-get update -qq || true
    apt-get install -y --no-install-recommends "${MISSING[@]}" || {
        echo "  apt-get install reported errors — re-checking our packages..."
        STILL_MISSING=()
        for p in "${MISSING[@]}"; do
            dpkg-query -W -f='${Status}' "$p" 2>/dev/null \
                | grep -q "install ok installed" || STILL_MISSING+=("$p")
        done
        if [ ${#STILL_MISSING[@]} -ne 0 ]; then
            echo "ERROR: still missing: ${STILL_MISSING[*]}"
            exit 1
        fi
        echo "  all our deps installed despite apt errors (likely an unrelated"
        echo "  broken package such as nvidia-dkms); continuing."
    }
fi

# ---------- 2. NVIDIA Video Codec headers (no NVIDIA login needed) ----------
echo
echo "[2/6] Installing nv-codec-headers (NVDEC / NVENC headers)..."
NVCODEC_DIR=$SRC_ROOT/nv-codec-headers
mkdir -p "$SRC_ROOT"
chown -R "$SUDO_USER_NAME":"$SUDO_USER_NAME" "$SRC_ROOT"
sudo -u "$SUDO_USER_NAME" git clone --depth 1 \
    https://github.com/FFmpeg/nv-codec-headers.git "$NVCODEC_DIR" 2>/dev/null \
    || (cd "$NVCODEC_DIR" && sudo -u "$SUDO_USER_NAME" git pull)
make -C "$NVCODEC_DIR" install PREFIX=/usr/local
ldconfig

echo "  installed:"
ls /usr/local/include/ffnvcodec/

# ---------- 3. OpenCV source (5.x branch, not a tag) ----------
echo
echo "[3/6] Cloning OpenCV $OPENCV_BRANCH branch + contrib..."
# If a stale opencv/ from a previous failed run exists with a different tag,
# wipe it so we always get a clean 5.x checkout.
sudo -u "$SUDO_USER_NAME" bash <<EOF
set -e
cd "$SRC_ROOT"
for d in opencv opencv_contrib; do
    if [ -d "\$d" ]; then
        cur_branch=\$(git -C "\$d" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
        if [ "\$cur_branch" != "$OPENCV_BRANCH" ]; then
            echo "  removing stale \$d (was on \$cur_branch, want $OPENCV_BRANCH)"
            rm -rf "\$d"
        fi
    fi
done
[ -d opencv ]         || git clone --branch $OPENCV_BRANCH --depth 1 https://github.com/opencv/opencv.git
[ -d opencv_contrib ] || git clone --branch $OPENCV_BRANCH --depth 1 https://github.com/opencv/opencv_contrib.git
echo "  opencv         HEAD: \$(git -C opencv rev-parse --short HEAD)"
echo "  opencv_contrib HEAD: \$(git -C opencv_contrib rev-parse --short HEAD)"
EOF

# ---------- 4. CMake configure ----------
echo
echo "[4/6] CMake configure..."
BUILD_DIR=$SRC_ROOT/opencv/build
sudo -u "$SUDO_USER_NAME" mkdir -p "$BUILD_DIR"

# Python bindings are NOT built — this project only consumes OpenCV from C++
# (TRT pipeline) and from the CUDA modules. Skipping Python avoids the
# OpenCV / Python 3.14 detection bug entirely.
echo "  python3       : DISABLED (C++ only build)"

# Build cmake invocation; cudacodec flags depend on whether SDK headers
# are present at the location OpenCV's WITH_NVCUVID check uses.
# Try VCSDK_INCLUDE env first; fall back to scanning $HOME/Downloads for
# Video_Codec_SDK_*/Interface; finally check if the headers were already
# copied to /usr/local/cuda/include by a previous run.
CUDACODEC_FLAGS=()
SDK_FOUND=""
if [ -n "${VCSDK_INCLUDE:-}" ] && [ -f "${VCSDK_INCLUDE}/nvcuvid.h" ]; then
    SDK_FOUND="$VCSDK_INCLUDE"
else
    # auto-discover under user's Downloads
    SDK_HOME=$(eval echo "~$SUDO_USER_NAME")/Downloads
    SDK_AUTO=$(ls -d "$SDK_HOME"/Video_Codec_SDK_*/Interface 2>/dev/null | head -1)
    if [ -n "$SDK_AUTO" ] && [ -f "$SDK_AUTO/nvcuvid.h" ]; then
        SDK_FOUND="$SDK_AUTO"
    fi
fi
if [ -z "$SDK_FOUND" ] && [ -f /usr/local/cuda/include/nvcuvid.h ]; then
    SDK_FOUND="(already staged at /usr/local/cuda/include)"
fi

if [ -n "$SDK_FOUND" ]; then
    echo "  Video Codec SDK headers found: $SDK_FOUND"
    if [ -d "$SDK_FOUND" ]; then
        cp -n "$SDK_FOUND"/nvcuvid.h "$SDK_FOUND"/cuviddec.h \
              "$SDK_FOUND"/nvEncodeAPI.h /usr/local/cuda/include/ 2>/dev/null || true
    fi
    CUDACODEC_FLAGS=(
        -DWITH_NVCUVID=ON
        -DWITH_NVCUVENC=ON
        -DBUILD_opencv_cudacodec=ON
    )
else
    echo "  Video Codec SDK NOT found; cudacodec will be DISABLED."
    echo "  Download from https://developer.nvidia.com/video-codec-sdk and"
    echo "  unzip into ~/Downloads/Video_Codec_SDK_*/  (auto-detected next run)"
    CUDACODEC_FLAGS=(
        -DWITH_NVCUVID=OFF
        -DWITH_NVCUVENC=OFF
        -DBUILD_opencv_cudacodec=OFF
    )
fi

# Force re-detection of Python3 by wiping the CMake cache. ninja artifacts
# (in CMakeFiles/) survive, so already-built C++ libs won't be rebuilt -
# only changed/new targets (e.g. cv2 Python module) will compile.
sudo -u "$SUDO_USER_NAME" rm -f "$BUILD_DIR/CMakeCache.txt"

sudo -u "$SUDO_USER_NAME" cmake -GNinja -S "$SRC_ROOT/opencv" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DOPENCV_EXTRA_MODULES_PATH="$SRC_ROOT/opencv_contrib/modules" \
    \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=OFF \
    -DOPENCV_DNN_CUDA=OFF \
    "${CUDACODEC_FLAGS[@]}" \
    \
    -DCUDA_ARCH_BIN="$CUDA_ARCH" \
    -DCUDA_ARCH_PTX="$CUDA_ARCH" \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCUDA_FAST_MATH=ON \
    \
    -DWITH_FFMPEG=ON \
    -DWITH_GSTREAMER=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_opencv_python_bindings_generator=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DOPENCV_ENABLE_NONFREE=ON

# Sanity-check the configure output
echo
echo "  ----- KEY CONFIGURE FLAGS -----"
grep -E "BUILD_opencv_cudacodec|WITH_NVCUVID|WITH_NVCUVENC|WITH_CUDA:BOOL|CUDA_ARCH_BIN" \
    "$BUILD_DIR/CMakeCache.txt" | head -10
if ! grep -q "WITH_CUDA:BOOL=ON" "$BUILD_DIR/CMakeCache.txt"; then
    echo "ERROR: CUDA support not enabled — aborting before the long build"
    exit 1
fi

# ---------- 5. Build ----------
echo
echo "[5/6] Building with $JOBS jobs (this is the slow part: ~30-50 min)..."
sudo -u "$SUDO_USER_NAME" ninja -C "$BUILD_DIR" -j "$JOBS"

# ---------- 6. Install ----------
echo
echo "[6/6] Installing to $INSTALL_PREFIX..."
ninja -C "$BUILD_DIR" install
ldconfig

# ---------- Verify ----------
echo
echo "============================================================"
echo "DONE. Verifying..."
echo "============================================================"
echo
echo "C++ symbol check:"
ls "$INSTALL_PREFIX/lib"/libopencv_cudacodec* 2>&1 | head -3
echo
echo "C++ test (compile a tiny CUDA-using program):"
TMP_TEST=/tmp/opencv5_test_$$.cc
cat > "$TMP_TEST" <<'CPPEOF'
#include <opencv2/core/cuda.hpp>
#include <iostream>
int main() {
    int n = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << n << "\n";
    if (n > 0) {
        cv::cuda::DeviceInfo info(0);
        std::cout << "Device 0: " << info.name() << " sm_"
                  << info.majorVersion() << info.minorVersion() << "\n";
    }
    return 0;
}
CPPEOF
PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig g++ -std=c++17 "$TMP_TEST" \
    -o /tmp/opencv5_test \
    $(PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig pkg-config --cflags --libs opencv5) \
    && /tmp/opencv5_test \
    || echo "  C++ test compile failed"
rm -f "$TMP_TEST" /tmp/opencv5_test

echo
echo "pkg-config check:"
PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig pkg-config --modversion opencv5
echo
echo "If pkg-config still reports the old apt version, your build.sh should"
echo "set:   export PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH"
