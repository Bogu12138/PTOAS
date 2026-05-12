# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PTOAS is an MLIR-based compiler toolchain for Huawei Ascend NPU tile-level operations. It lowers the PTO (Programming Tiling Operator) dialect — which models Ascend's Cube/Vector/MTE pipelines, tile buffers, synchronization primitives, and inter-core communication — through a pass pipeline to C++ code via the EmitC dialect.

Built as an **out-of-tree** MLIR project against **LLVM llvmorg-19.1.7** (commit `cd708029e0b2869e80abe31ddb175f7c35361f90`). C++17 required.

## Build Commands

```bash
# Configure (requires pre-built LLVM/MLIR)
cmake -G Ninja -S . -B build \
  -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
  -DPython3_EXECUTABLE=$(which python3) \
  -DPython3_FIND_STRATEGY=LOCATION \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core \
  -DCMAKE_INSTALL_PREFIX=install

# Build CLI tools and Python bindings
ninja -C build ptoas ptobc
ninja -C build install

# Build everything (includes Python extension)
ninja -C build
```

## Test Commands

```bash
# Run lit regression tests
ninja -C build check-pto

# Run a single lit test
build/tools/ptoas/ptoas test/lit/pto/some_test.pto | FileCheck test/lit/pto/some_test.pto

# Run sample end-to-end tests (Python -> .pto -> .cpp)
bash test/samples/runop.sh --enablebc all

# Run a single sample
cd test/samples/MatMul && python3 tmatmulk.py | build/tools/ptoas/ptoas - -o tmatmulk.cpp
```

## Running ptoas

```bash
# Parse and print PTO IR
ptoas input.pto

# Compile to C++ (with sync insertion, for A3)
ptoas input.pto --enable-insert-sync --pto-arch=a3 -o output.cpp

# Alternative sync strategies
ptoas input.pto --enable-graph-sync-solver -o output.cpp
ptoas input.pto --enable-inject-barrier-all-sync -o output.cpp

# Level3 skips PlanMemory/InsertSync
ptoas input.pto --pto-level=level3 -o output.cpp

# Binary bytecode input (auto-detected by magic bytes "PTOBC\0")
ptoas input.ptobc -o output.cpp
```

## Architecture

### Compilation Pipeline (in `tools/ptoas/ptoas.cpp`)

```
.pto (text MLIR) / .ptobc (binary) -> Parse -> ModuleOp
  -> AssignDefaultFrontendPipeId
  -> LowerFrontendPipeOps
  -> InferValidatePipeInit
  -> LoweringSyncToPipe
  -> InferPTOLayout
  -> PTOA5NormalizeTMov
  -> PTOViewToMemref
  -> PlanMemory
  -> ResolveReservedBuffers
  -> SyncInsertion (one of three strategies)
  -> MaterializeTileHandles
  -> CSE
  -> PTOToEmitC lowering
  -> FormExpressions + CSE
  -> C++ output (via emitc::translateToCpp + post-processing)
```

### Key Directories

| Path | Purpose |
|---|---|
| `include/PTO/IR/*.td` | ODS/TableGen definitions for all ops, types, attrs, enums, interfaces |
| `include/PTO/IR/*.h` | Public C++ headers; includes generated `.inc` files |
| `lib/PTO/IR/` | Dialect registration, type/attr parsers, verifiers |
| `lib/PTO/Transforms/` | All compiler passes (sync, memory planning, lowering, EmitC) |
| `tools/ptoas/ptoas.cpp` | Main CLI driver (~1250 lines): pipeline construction, C++ post-processing |
| `tools/ptobc/` | Binary bytecode encoder/decoder (`ptobc encode` / `ptobc decode`) |
| `python/pto/dialects/` | Python bindings: generated `_pto_ops_gen.py` + hand-written `pto.py` |
| `lib/Bindings/Python/` | pybind11 native extension (`_pto`) |
| `test/lit/pto/*.pto` | Lit regression tests (~90+ files) using `ShTest` + `FileCheck` |
| `test/samples/` | End-to-end Python sample programs (~90+ directories) |

### Key Types and Concepts

- **Hardware targets**: A3 (`Ascend910B1`) and A5 (`Ascend950`) — controlled via `--pto-arch`
- **Pipes**: S, V, M, MTE1–MTE5 — represent hardware execution pipelines
- **Tile/TileBuf**: Hardware tile abstractions with memory space, valid shape, config
- **Sync**: `set_flag`/`wait_flag`/`barrier` for intra-core; `sync_set`/`sync_wait` for inter-core
- **Memory spaces**: GM (global), MAT, LEFT, RIGHT, ACC, VEC, BIAS, SCALING

### ODS/TableGen Structure

`PTOOps.td` is the master file that includes `PTODialect.td`, `PTOAttrs.td`, `PTOTypeDefs.td`, and `PTOInterfaces.td`. TableGen generates `.h.inc` and `.cpp.inc` files for the dialect, ops, types, attrs, enums, and interfaces.

## Cross-Layer Synchronization

When changing any user-visible behavior, keep these layers synchronized:

1. **ODS definitions**: `include/PTO/IR/*.td` — op definitions, types, attrs, assembly format
2. **C++ implementation**: `lib/PTO/IR/` (verifiers), `lib/PTO/Transforms/` (lowering/passes)
3. **CLI tool**: `tools/ptoas/ptoas.cpp` — flags, pipeline construction, C++ post-processing
4. **Python bindings** (if affected): `python/pto/dialects/pto.py`, `lib/Bindings/Python/`
5. **Docs**: `README.md`, `docs/`
6. **Tests**: `test/lit/`, `test/samples/`

See `.claude/rules/cross-layer-sync.md` for the detailed checklist.

## File Headers

All source/script files must include the PR386 OAT.3 license header:
```
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// ...
```
When touching an existing file that lacks this header, add it in the same change.

## Testing Guidelines

- **Lit tests** (`test/lit/pto/*.pto`): For IR/lowering/codegen regressions. Use `FileCheck` with `// CHECK:` directives.
- **Sample tests** (`test/samples/`): End-to-end Python-to-C++ tests via `runop.sh`.
- For pass bugs, provide: minimal input IR, exact `ptoas` invocation, and expected property.
- Keep reproductions minimal. Avoid committing temporary scripts outside `test/`.

See `.claude/rules/testing-and-examples.md` for details.
