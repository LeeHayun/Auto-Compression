#/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

mkdir -p build/local

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

CMAKE_ARGS+=("-DXNNPACK_LIBRARY_TYPE=static")

CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")

CMAKE_ARGS+=("-DXNNPACK_BUILD_ODROID=ON")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/local && cmake ../.. \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ "$(uname)" == "Darwin" ]
then
  cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
  cmake --build . -- "-j$(nproc)"
fi
