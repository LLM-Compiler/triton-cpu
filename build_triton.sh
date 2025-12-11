#!/bin/bash
# set -x
# Build triton-cpu from source with custom LLVM
# Tharindu Patabandi <tharindu@protonmail.com>

# TODO: automate llvm detection
export LLVM_BUILD_DIR=../llvm-project/build
echo "LLVM_BUILD_DIR=$(realpath "$LLVM_BUILD_DIR")"

echo "Building Triton..."

if [[ -d $LLVM_BUILD_DIR && -n "$(ls "$LLVM_BUILD_DIR")" ]]; then

	LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include" \
	LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib" \
	LLVM_SYSPATH="$LLVM_BUILD_DIR" \
	TRITON_BUILD_WITH_CLANG_LLD=true \
	TRITON_BUILD_WITH_CCACHE=true \
	pip install -v -e ./python --no-build-isolation
else
	echo "Triton could not find LLVM build. Exiting!"
fi
