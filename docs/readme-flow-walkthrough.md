# PTOAS 全流程走读：README 章节 1–5.3

本文档对 README 中的 1–5.3 节进行端到端代码走读，串通从项目理解、构建、运行环境配置到 CLI/Python 使用和测试的完整流程。

> 5.4 上板验证需要 NPU 硬件，不在本文档范围内。

---

## 1. 项目简介

PTOAS 是基于 **LLVM/MLIR (llvmorg-19.1.7)** 构建的编译器工具链，目标是为华为昇腾 NPU 的 PTO Bytecode 提供：

- IR 解析与验证
- 编译优化 Pass（算子融合、同步插入等）
- 代码生成（PTO IR → EmitC → C++）
- Python 绑定（PyPTO、TileLang 等框架直接构建 IR）

架构模式：**Out-of-Tree** MLIR 项目，不修改 LLVM 源码。

---

## 2. 目录结构

```
PTOAS/
├── include/PTO/IR/           # ODS/TableGen 定义 (.td) + C++ 头文件 (.h)
│   ├── PTOOps.td             # 主入口：包含所有 op 定义（~50K tokens）
│   ├── PTOAttrs.td           # 属性和枚举定义
│   ├── PTOTypeDefs.td        # 类型定义
│   ├── PTOInterfaces.td      # Op 接口
│   └── PTODialect.td         # 方言注册
├── lib/
│   ├── PTO/
│   │   ├── IR/               # 方言注册、类型/属性解析器、验证器
│   │   └── Transforms/       # 所有编译 Pass
│   ├── CAPI/                 # C API 暴露
│   └── Bindings/Python/      # pybind11 原生扩展 (_pto)
├── python/
│   └── pto/dialects/
│       ├── pto.py            # 手写 Python 绑定（高层封装）
│       └── PTOOps.td         # ODS 定义（用于生成 _pto_ops_gen.py）
├── tools/
│   ├── ptoas/                # ptoas CLI 入口（~1250 行）
│   └── ptobc/                # ptobc CLI 入口（编解码二进制 bytecode）
├── test/
│   ├── lit/pto/              # Lit/FileCheck 回归测试（~90+ 文件）
│   ├── samples/              # 端到端 Python 样例程序（~90+ 目录）
│   └── npu_validation/       # NPU 上板验证基础设施
└── CMakeLists.txt            # 根构建配置（版本 0.38，C++17）
```

### 关键文件速查

| 你要改什么 | 去哪里改 |
|---|---|
| Op 定义 / 汇编格式 | `include/PTO/IR/PTOOps.td` |
| 类型 / 属性 / 枚举 | `include/PTO/IR/PTOTypeDefs.td`, `PTOAttrs.td` |
| C++ 验证逻辑 | `lib/PTO/IR/` |
| 编译 Pass 实现 | `lib/PTO/Transforms/` |
| Pass 执行顺序 / CLI 标志 | `tools/ptoas/ptoas.cpp` (1143–1214 行) |
| Python 高层 API | `python/pto/dialects/pto.py` |
| C++ post-processing | `tools/ptoas/ptoas.cpp` (1221–1257 行) |

---

## 3. 构建指南

### 3.0 环境要求

| 依赖 | 版本要求 |
|---|---|
| OS | Linux (Ubuntu 20.04+)，CI 使用 ubuntu-22.04 |
| 编译器 | GCC >= 9 或 Clang，C++17 |
| CMake | >= 3.20 |
| Ninja | 任意版本 |
| Python | 3.8+，需 `pybind11 < 3` + `numpy` |
| LLVM/MLIR | **严格 llvmorg-19.1.7** (commit `cd708029e0b2869e80abe31ddb175f7c35361f90`) |

> macOS ARM64 可以构建 LLVM/MLIR，但 PTOAS 未在 macOS 上测试过。CI 仅在 Linux 上运行。

### 3.1 构建 LLVM/MLIR（依赖）

```bash
export WORKSPACE_DIR=$HOME/llvm-workspace
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

cd $WORKSPACE_DIR
git clone https://github.com/llvm/llvm-project.git
cd $LLVM_SOURCE_DIR
git checkout llvmorg-19.1.7

cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

ninja -C $LLVM_BUILD_DIR
```

### 3.2 构建 PTOAS

```bash
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install
export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)

cd $PTO_SOURCE_DIR
cmake -G Ninja -S . -B build \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DPython3_EXECUTABLE=$(which python3) \
    -DPython3_FIND_STRATEGY=LOCATION \
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core \
    -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

ninja -C build ptoas ptobc    # 构建 CLI 工具
ninja -C build install         # 安装 Python 方言文件 + 原生扩展
```

### 3.3 构建产物

```
build/tools/ptoas/ptoas              # ptoas CLI 可执行文件
build/tools/ptobc/ptobc              # ptobc CLI 可执行文件
build/python/pto/dialects/           # 生成的 Python 绑定
install/mlir/dialects/               # 安装的 Python 方言文件
$LLVM_BUILD_DIR/.../mlir/_mlir_libs/ # 安装到 MLIR Python 包的原生扩展
```

---

## 4. 运行环境配置

构建完成后，设置以下环境变量：

```bash
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH

export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH

export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PTO_SOURCE_DIR/build/tools/ptobc:$PATH
```

---

## 5. 使用方法

### 5.1 CLI 工具 (ptoas)

`ptoas` 是主编译器驱动，代码入口在 `tools/ptoas/ptoas.cpp:919`。

#### 基本用法

```bash
# 解析并打印 PTO IR（仅验证，不编译）
ptoas input.pto

# 编译为 C++，启用同步插入
ptoas input.pto --enable-insert-sync -o output.cpp

# 指定目标架构（A3 / A5）
ptoas input.pto --pto-arch=a5 -o output.cpp

# Level3 跳过 PlanMemory 和 InsertSync（需要手动指定地址）
ptoas input.pto --pto-level=level3 -o output.cpp

# 查看版本
ptoas --version
```

#### 输入格式

- **文本 MLIR (.pto)**：标准的 MLIR 文本格式，由 MLIR 解析器处理
- **二进制 bytecode (.ptobc)**：通过魔术字节 `"PTOBC\0"` 自动检测，由 `ptobc::decodePTOBCToModule()` 解码

#### 三种同步策略（互斥）

| 标志 | 对应 Pass | 说明 |
|---|---|---|
| `--enable-insert-sync` | `PTOInsertSync` | 基于数据流依赖的 set/wait_flag 插入 |
| `--enable-inject-barrier-all-sync` | `PTOInjectBarrierAllSync` | 保守策略：所有 pipe 间插入 barrier |
| `--enable-graph-sync-solver` | `PTOGraphSyncSolver` | 基于图求解的同步优化 |

#### 编译 Pipeline 代码位置

Pipeline 在 `tools/ptoas/ptoas.cpp:1138–1214` 构建，按顺序：

```
1. AssignDefaultFrontendPipeId      → 1143 行
2. LowerFrontendPipeOps             → 1145 行
3. InferValidatePipeInit            → 1148 行
4. LoweringSyncToPipe               → 1149 行
5. InferPTOLayout                   → 1152 行（可用 --disable-infer-layout 跳过）
6. A5NormalizeTMov                  → 1153 行
7. PTOViewToMemref                  → 1154 行
8. PlanMemory                       → 1157 行（level3 跳过）
9. ResolveReservedBuffers           → 1163 行
10. SyncInsertion（三选一）          → 1168–1178 行
11. MaterializeTileHandles          → 1206 行
12. CSE                             → 1207 行
13. PTOToEmitC（架构相关）           → 1208–1212 行
14. FormExpressions + CSE           → 1213–1214 行
```

之后进入 C++ 发射和后处理（1221–1257 行）：
- `reorderEmitCFunctions()` — 函数拓扑排序
- `emitc::translateToCpp()` — EmitC → C++ 文本
- Post-processing：重写 marker 调用、指针下标、event-id 数组下标、标量常量折叠、GlobalTensor null-init 修复

#### ptobc 工具

```bash
# 编码：文本 .pto → 二进制 .ptobc
ptobc encode input.pto -o output.ptobc

# 解码：二进制 .ptobc → 文本 .pto
ptobc decode input.ptobc -o output.pto
```

### 5.2 Python 接口

#### 为什么需要 Python 绑定

PTOAS 的 Python 绑定服务于一个明确目标：

> PyPTO、TileLang、CuTile 等框架在 Python 端直接构建、操作和编译 PTO Bytecode

具体原因：

1. **上层框架都是 Python 生态**。AI 编程模型（PyTorch、JAX 等）和昇腾上层框架都在 Python 中工作。要让这些框架生成 PTO IR，必须有 Python 构造接口。`tmatmulk.py` 本身就是一个例子——用 Python 描述 kernel，比手写 `.pto` 文本高效得多。

2. **快速原型验证**。Python 比 C++ 更灵活，写一个 MatMul kernel 用 Python ~270 行，可以 `build(M=64, K=512, N=128)` 参数化调参，而手写 `.pto` 文本则无法参数化。

3. **与 MLIR Python 生态集成**。MLIR 上游 dialect（`func`, `arith`, `scf` 等）都有 Python 绑定，PTO 作为 out-of-tree dialect 通过同一机制暴露，使得 `from mlir.dialects import pto, func, arith, scf` 可以无缝混用。

##### 这是 MLIR 的主流做法吗

**是的。** Python 绑定是 MLIR 的一等公民特性，不是附加功能。

| 项目 | Python 绑定 | 用途 |
|---|---|---|
| **MLIR 上游** (Linalg, Tensor, Func...) | 官方内置 (`mlir-python-bindings`) | 所有标准 dialect 都有 Python API |
| **IREE** (OpenXLA 运行时) | 大量 Python | 编译器驱动、测试、模型导入 |
| **torch-mlir** | 核心就是 Python | 把 PyTorch 模型转成 MLIR |
| **StableHLO / JAX** | Python 为主 | XLA 编译器的 Python 前端 |
| **PTOAS** | Python 绑定 | 上层 AI 框架 → PTO IR 构造 |

MLIR 的 Python 绑定架构是统一的：

```
C++ Dialect 定义 (.td)
    ↓ mlir-tablegen -gen-python-op-bindings
自动生成的 Python ops (_pto_ops_gen.py)
    ↓ + 手写封装 (pto.py)
Python 用户 API
```

PTOAS 的做法完全遵循这个标准模式。需要强调的是：**Python 不是"实现"，是绑定（binding）**。核心编译逻辑全在 C++（`lib/PTO/`），Python 只构造 IR 然后交给 `ptoas` 编译。

#### Python 绑定的三层结构

##### 层级 1：原生扩展 (`_pto`)

C++ pybind11 模块，位于 `lib/Bindings/Python/PTOModule.cpp`。注册 PTO 方言、暴露类型和属性。

##### 层级 2：自动生成绑定 (`_pto_ops_gen.py`)

由 TableGen 从 `PTOOps.td` 自动生成，包含所有 Op 的 Python 构造器。

##### 层级 3：手写封装 (`pto.py`)

位于 `python/pto/dialects/pto.py`（~724 行），在生成绑定之上提供高层 API：

```python
from mlir.ir import Context, Module, Location
from mlir.dialects import pto

with Context() as ctx, Location.unknown():
    pto.register_dialect(ctx, load=True)
    module = Module.create()
    # 使用 pto.xxx 构建IR...
    print(module)  # 输出 .pto 格式的文本 MLIR
```

#### 高层 API 封装一览

| 封装函数 | 底层 op | 用途 |
|---|---|---|
| `pto.record_event()` | `pto.set_flag` | 管道间同步：记录事件 |
| `pto.wait_event()` | `pto.wait_flag` | 管道间同步：等待事件 |
| `pto.barrier()` | `pto.set_flag` + `pto.wait_flag` | 全管道屏障 |
| `pto.sync_set()` / `pto.sync_wait()` | `pto.sync_set` / `pto.sync_wait` | 核间同步 |
| `pto.get_buf()` / `pto.rls_buf()` | `pto.get_buf` / `pto.rls_buf` | A5 缓冲区同步 |
| `pto.load_scalar()` / `pto.store_scalar()` | 标量加载/存储 | 标量指针操作 |
| `pto.TileConfig` | — | 硬件 tile 尺寸常量 |

#### 典型 Python 构建流程

以 `test/samples/MatMul/tmatmulk.py` 为例：

```python
# 1. 注册方言
pto.register_dialect(ctx, load=True)

# 2. 创建 Module 并设置设备属性
module = builtin.ModuleOp()
module.attributes["pto.device-spec"] = StringAttr.get("Ascend910B1")

# 3. 定义类型
ptr_a = pto.PtrType.get(F32Type.get())
tile_buf = pto.TileBufType.get([32, 32], F32Type.get(),
    pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT),
    [32, 32], cfg)

# 4. 构建 IR
tvA = pto.MakeTensorViewOp(tv_type, a_ptr, [cM, cK], [cK, c1]).result
aTile = pto.AllocTileOp(tile_buf).result
pto.TLoadOp(None, svA, aMatTile)
pto.TMovOp(None, aMatTile, aTile)
pto.TMatmulOp(None, aTile, bTile, cTile)
pto.TStoreOp(None, cTile, svOut)

# 5. 输出
print(module)  # → .pto 文本
```

### 5.3 运行测试

#### Lit 回归测试

Lit 测试位于 `test/lit/pto/`，使用 `ShTest` + `FileCheck` 格式：

```bash
# 运行全部 lit 测试
ninja -C build check-pto

# 运行单个 lit 测试
build/tools/ptoas/ptoas test/lit/pto/empty_func.pto | FileCheck test/lit/pto/empty_func.pto
```

测试文件格式示例 (`test/lit/pto/empty_func.pto`)：

```mlir
// RUN: ptoas %s | FileCheck %s

module {
  func.func @hello() -> () {
    return
  }
}

// CHECK: __global__ AICORE void hello() {
```

测试覆盖：plan_memory、sync（insert/graph/barrier）、subview、materialize_tile_handles、EmitC 发射、解析兼容性、低精度类型、异步 DMA、通信 op、内核入口验证、issue 回归测试等。

#### Python 样例测试

样例位于 `test/samples/`，每个子目录是一个独立测试：

```bash
# 运行全部样例（含 ptobc 往返）
bash test/samples/runop.sh --enablebc all

# 运行单个样例
bash test/samples/runop.sh -t MatMul

# 手动运行单个样例
cd test/samples/MatMul
python3 tmatmulk.py > tmatmulk.pto
ptoas tmatmulk.pto --enable-insert-sync -o tmatmulk.cpp
```

`runop.sh` 的流程：

```
.py → python3 → .pto → ptoas → .cpp
                   ↓ (如果 --enablebc)
               ptobc encode → .ptobc → ptobc decode → .pto → ptoas → .cpp
```

环境变量控制：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PTOAS_BIN` | 自动检测 | ptoas 可执行文件路径 |
| `PTOBC_BIN` | 自动检测 | ptobc 可执行文件路径 |
| `PYTHON_BIN` | `python3` | Python 解释器 |
| `PTOAS_OUT_DIR` | 临时目录 | 输出目录 |
| `PTOAS_ENABLE_INSERT_SYNC` | `1` | 是否追加 `--enable-insert-sync` |
| `PTOAS_FLAGS` | 空 | 传递给 ptoas 的额外参数 |

---

## 端到端数据流总结

```
  Python 脚本 (test/samples/MatMul/tmatmulk.py)
    │  import mlir.dialects.pto
    │  构建 PTO IR (AllocTile, TLoad, TMov, TMatmul, TStore, sync ops)
    ▼
  .pto 文本 (tmatmulk.pto)
    │  MLIR 文本格式，包含 PTO 方言 ops
    ▼
  ptoas CLI (tools/ptoas/ptoas.cpp)
    │  解析 → 方言注册 → 架构检测 (A3/A5)
    │  Pass Pipeline:
    │    FrontendPipe → SyncToPipe → LayoutInfer → ViewToMemref
    │    → PlanMemory → SyncInsert → MaterializeTile → EmitC
    │  C++ 后处理 (marker 重写, 下标, 折叠)
    ▼
  C++ kernel (tmatmulk.cpp)
    │  #include "pto/pto-inst.hpp"
    │  __global__ AICORE void RunTMATMULSplitK(...)
    │  Tile<...> 声明, TLOAD, TMOV, TMATMUL, TSTORE 等
    ▼
  [昇腾硬件编译器] → AICORE binary → [NPU 执行]   ← 5.4 上板验证，需 NPU
```

---

## 为什么需要 MLIR 编译层

C++ kernel 看起来比 Python 和 .pto 都简明，但这恰恰说明 MLIR 层的价值——复杂性被正确地吸收到了编译器中。

### C++ 的简明是编译的结果，不是起点

对比同一操作的三个表示：

```python
# Python: 描述 "做什么"（可参数化）
pto.AllocTileOp(tile_buf_aMat)
pto.TLoadOp(None, svA, aMatTile)
pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
```

```mlir
% .pto: 结构化中间表示
%4 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, ...>
pto.tload ins(%13) outs(%4)
pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
```

```cpp
// C++: 最终产物，包含所有 "怎么做" 的细节
Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32,
     SLayout::RowMajor, 512, PadValue::Null> v16;
TASSIGN(v16, v13);   // 地址 0（由 PlanMemory 计算得出）
TLOAD(v16, v35);
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
```

C++ 看起来简明，是因为 MLIR pipeline 已经替你完成了以下工作：

| 隐藏的复杂性 | 由哪个 Pass 处理 | 手写 C++ 需要做什么 |
|---|---|---|
| Tile 地址分配 | PlanMemory | 手算每个 tile 的内存偏移量（0, 4096, 8192...） |
| 同步点插入 | PTOInsertSync / GraphSyncSolver | 分析数据依赖，在正确位置插入 set/wait_flag |
| 指针偏移计算 | ViewToMemref + PTOToEmitC | 手算 `__gm__ float* v32 = v2 + v31` 中的每个偏移 |
| GlobalTensor 模板参数 | PTOToEmitC | 手动确定 `Shape<1,1,1,32,32>` 和 `Stride<1024,...>` |
| 分形布局适配 | InferPTOLayout | 理解 ND/DN/NZ 布局规则并应用 |

**如果直接写 C++**，需要手动完成上面所有计算，且无法参数化（换 M=64 就要重新手算所有地址和偏移）。

### 主流生态中的定位

这和 AI 编译器领域的做法完全一致：

**PyTorch → torch-mlir → Linalg → CUDA/HIP**：Python 描述网络结构，编译器做 fusion/tiling/memory planning，生成看起来简单但包含所有优化细节的 CUDA kernel。

**JAX → HLO → XLA → PTX**：Python 描述计算，编译器做 autodiff/融合/布局优化，生成机器码。

PTOAS 遵循同样的模式：

```
Python (PyPTO/TileLang)     用户只描述 tile 级操作
  ▼
PTO MLIR (.pto)             编译器做 sync/memory plan/layout/lowering
  ▼
C++ kernel (pto-isa)        看起来简单，但包含了所有优化后的细节
```

### MLIR 层的具体价值

**1. 优化 Pass 可组合**

想换同步策略？一个 flag 就行，同一个 `.pto` 输入可以生成不同的同步方案：

```bash
ptoas input.pto --enable-insert-sync              # 数据流同步
ptoas input.pto --enable-graph-sync-solver         # 图求解同步
ptoas input.pto --enable-inject-barrier-all-sync   # 保守屏障
```

**2. 多目标支持**

```bash
ptoas input.pto --pto-arch=a3 -o a3.cpp   # Ascend 910B1
ptoas input.pto --pto-arch=a5 -o a5.cpp   # Ascend 950
```

A3 和 A5 的 tile 配置、同步语义、指令集都不同。MLIR IR 是硬件无关的，lowering 时才分化。

**3. 正确性验证**

`.pto` 中的类型系统（`tile_buf<loc=left, ...>` 只能传给 `tmov` 的对应位置）在 IR 层做验证。手写 C++ 时，传错 tile 给错误的位置不会报错，运行时才崩。

**4. 上层框架集成**

PyPTO、TileLang、CuTile 等 Python 框架直接构造 PTO IR，然后统一编译。没有 MLIR 层，每个框架都要自己生成 C++，重复实现内存规划、同步插入等逻辑。

### 一句话总结

**MLIR 层的价值不在于生成更可读的 C++，而在于让用户只需描述"做什么"，由编译器自动推导"怎么做"。**

---

## 当前环境状态

| 项目 | 状态 |
|---|---|
| 本机平台 | macOS 15.6 ARM64, Apple Clang 17, 10 核, 16GB RAM |
| LLVM/MLIR 源码 | 未下载 |
| PTOAS 构建 | 未构建 |
| Python 绑定 | 未构建 |
| Lit 测试 | 无法运行（需 ptoas 二进制） |
| 样例测试 | 无法运行（需 ptoas + LLVM Python） |

如需实际构建运行，建议使用 Linux 环境（Docker / WSL / 虚拟机），参考 `docker/Dockerfile` 或 CI 配置 `.github/workflows/ci.yml`。
