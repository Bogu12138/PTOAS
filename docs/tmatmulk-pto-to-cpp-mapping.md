# tmatmulk.pto → tmatmulk.cpp 编译流程逐层分析

本文档追踪 `tmatmulk.pto`（PTO IR）经过 `ptoas` 编译 pipeline 转换为 `tmatmulk.cpp`（C++ kernel）的每个步骤。

---

## 编译命令

```bash
ptoas tmatmulk.pto --enable-insert-sync -o tmatmulk.cpp
```

---

## Pipeline 总览

```
tmatmulk.pto (PTO IR)
  │
  │ 1. AssignDefaultFrontendPipeId
  │ 2. LowerFrontendPipeOps
  │ 3. InferValidatePipeInit
  │ 4. LoweringSyncToPipe       ← .pto 中已完成（set_flag/wait_flag）
  │ 5. InferPTOLayout
  │ 6. A5NormalizeTMov
  │ 7. PTOViewToMemref          ← TensorView → MemRef
  │ 8. PlanMemory               ← 分配地址，替换 alloc_tile
  │ 9. ResolveReservedBuffers
  │ 10. PTOInsertSync           ← --enable-insert-sync 模式下
  │ 11. MaterializeTileHandles  ← tile_buf → 实际指针
  │ 12. CSE
  │ 13. PTOToEmitC              ← PTO ops → EmitC ops（核心）
  │ 14. FormExpressions + CSE
  │
  │ Post-processing:
  │   - translateToCpp()         ← EmitC → C++ 文本
  │   - rewriteTileGetSetValueMarkers
  │   - rewriteScalarConstantDecls
  │   - rewriteHoistedGlobalTensorDecls
  │
  ▼
tmatmulk.cpp (C++ kernel)
```

---

## 1. 输入：tmatmulk.pto 结构

`.pto` 文件包含 64 行 PTO IR，核心结构：

```mlir
module attributes {"pto.device-spec" = "Ascend910B1"} {
  func.func @RunTMATMULSplitK(%arg0: !pto.ptr<f32>, ...) {
    // 常量定义 (10 个 arith.constant)
    // TensorView 创建 (4 个 pto.make_tensor_view)
    // Tile 分配 (7 个 pto.alloc_tile)
    scf.for ... {
      // PartitionView 切分 (3 个 pto.partition_view)
      // TLOAD (2 个 + 条件 bias)
      // set_flag / wait_flag (MTE2→MTE1)
      // TMOV (2 个 + 条件 bias)
      // set_flag / wait_flag (MTE1→M)
      // 条件 TMATMUL / TMATMUL_BIAS / TMATMUL_ACC
      // set_flag / wait_flag (M→MTE2)
    }
    // set_flag / wait_flag (M→FIX)
    // PartitionView + TSTORE
    return
  }
}
```

---

## 2. Pass 1–4：前端处理（已在 .pto 中完成）

`.pto` 文件已经经过以下 pass 的处理：

- **AssignDefaultFrontendPipeId**：为缺少 pipe_id 的 op 分配默认值 0
- **LowerFrontendPipeOps**：`record_event`/`wait_event` → `set_flag`/`wait_flag`（见 py→pto 文档）
- **InferValidatePipeInit**：验证 pipe 初始化配置
- **LoweringSyncToPipe**：SyncOpType → PIPE 映射（TLOAD→MTE2, TMOV_M2L→MTE1, TMATMUL→M）

---

## 3. Pass 5：InferPTOLayout

推断 GlobalTensor 的布局（ND/DN/NZ fractal 布局）。对于 MatMul 的行主序输入，推断为 ND（标准布局）。

---

## 4. Pass 7：PTOViewToMemref

将 PTO 的 TensorView / PartitionView 转换为 MLIR MemRef：

```
pto.make_tensor_view  →  memref + offset/stride 信息
pto.partition_view    →  memref.subview
```

这是必要的，因为后续的 PlanMemory 和 EmitC 阶段基于 MemRef 工作。

---

## 5. Pass 8：PlanMemory

**关键 Pass**。为每个 `alloc_tile` 分配本地内存地址。

### 转换前

```mlir
%4 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, ...>
```

### 转换后

PlanMemory 计算每个 tile 在本地内存中的偏移量，将地址信息附加到 op 上。地址通过 `TASSIGN` 宏在 C++ 中设置。

### C++ 输出中的对应

```cpp
Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32,
     SLayout::RowMajor, 512, PadValue::Null> v16;
TASSIGN(v16, v13);   // v13 = 0（基地址偏移）
```

**映射**：`TASSIGN(tile_var, address_offset)` 将 tile 绑定到本地内存地址。

---

## 6. Pass 10：PTOInsertSync（由 --enable-insert-sync 触发）

此 pass 分析数据流依赖，在需要时自动插入 `set_flag`/`wait_flag`。

但在本例中，`.pto` 中已包含完整的同步操作，因此此 pass 主要验证现有同步的正确性，不额外插入新的同步点。

---

## 7. Pass 11：MaterializeTileHandles

将逻辑上的 tile_buf 句柄转换为实际的内存指针操作，使 tile 可以被 EmitC 阶段引用。

---

## 8. Pass 13：PTOToEmitC（核心转换）

**这是最重要的 pass**，将所有 PTO ops 转换为 EmitC dialect ops。代码在 `lib/PTO/Transforms/PTOToEmitC.cpp`（~11890 行）。

### 转换机制

每个 PTO op 都有一个对应的 `OpConversionPattern`，将 PTO op 转换为 `emitc::CallOpaqueOp`（不透明的 C 函数调用）：

```
PTO op → OpConversionPattern::matchAndRewrite() → emitc::CallOpaqueOp
```

### 8.1 alloc_tile → Tile<...> 类型声明

**源码**：`lib/PTO/Transforms/PTOToEmitC.cpp:10535`

`.pto` 中的类型信息：

```mlir
%4 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32,
     v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=512, pad=0>
```

`getEmitCTileTypeString()` 函数（行 303）将 TileBufType 编码为 C++ 模板参数：

```cpp
"Tile<" + tileRoleToken + ", " + scalarType + ", "
         + rows + ", " + cols + ", "
         + blayout + ", " + vrow + ", " + vcol + ", "
         + slayout + ", " + fractal + ", " + pad + ">"
```

**映射表**：

| .pto 字段 | C++ 模板参数 | 值 |
|---|---|---|
| `loc=mat` | `TileType::Mat` | `tileRoleToken()` (行 260) |
| `dtype=f32` | `float` | 类型转换器 (行 360) |
| `rows=32` | `32` | 直接映射 |
| `cols=32` | `32` | 直接映射 |
| `blayout=col_major` | `BLayout::ColMajor` | `tileBufBLayoutToken()` (行 3617) |
| `v_row=32` | `32` | 直接映射 |
| `v_col=32` | `32` | 直接映射 |
| `slayout=row_major` | `SLayout::RowMajor` | `tileBufSLayoutToken()` (行 3626) |
| `fractal=512` | `512` | 直接映射 |
| `pad=0` | `PadValue::Null` | `tileBufPadToken()` (行 3637) |

**C++ 输出**：

```cpp
Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32,
     SLayout::RowMajor, 512, PadValue::Null> v16;
TASSIGN(v16, v13);  // v13 = 0（地址偏移）
```

### 7 个 Tile 的完整映射

| .pto SSA | .pto loc | C++ 变量 | C++ TileType | 地址偏移 |
|---|---|---|---|---|
| `%4` | mat | `v16` | `TileType::Mat, BLayout::ColMajor` | `v13` (0) |
| `%5` | mat | `v17` | `TileType::Mat, BLayout::ColMajor` | `v14` (4096) |
| `%6` | mat | `v18` | `TileType::Mat, BLayout::RowMajor, SLayout::NoneBox` | `v15` (8192) |
| `%7` | left | `v19` | `TileType::Left, BLayout::RowMajor` | `v13` (0) |
| `%8` | right | `v20` | `TileType::Right, BLayout::RowMajor, SLayout::ColMajor` | `v13` (0) |
| `%9` | acc | `v21` | `TileType::Acc, fractal=1024` | `v13` (0) |
| `%10` | bias | `v22` | `TileType::Bias, BLayout::RowMajor, SLayout::NoneBox` | `v13` (0) |

**注意**：LEFT/RIGHT/ACC/BIAS 的地址偏移都是 0（`v13`），因为 PlanMemory 为它们分配了独立的内存区域，复用同一基址偏移。MAT tiles 有不同偏移（0, 4096, 8192），因为它们共享 MAT 地址空间。

---

### 8.2 make_tensor_view → GlobalTensor 构造

**源码**：`buildGlobalTensorFromMemref()` 函数

`.pto` 中的 TensorView 经过 ViewToMemref pass 后变成 MemRef，PTOToEmitC 将 MemRef 转换为 C++ `GlobalTensor` 对象。

`.pto`：

```mlir
%0 = pto.make_tensor_view %arg1, shape = [%c32, %c256], strides = [%c256, %c1]
     : !pto.tensor_view<2xf32>
```

**C++ 输出**（TLOAD 内部的 GlobalTensor 构造）：

```cpp
__gm__ float* v32 = v2 + v31;    // 指针偏移计算
using GTShape_... = pto::Shape<1, 1, 1, 32, 32>;
using GTStride_... = pto::Stride<1024, 1024, 1024, 32, 1>;
GTShape_... v33 = GTShape_<...>();
GTStride_... v34 = GTStride_<...>();
using GT_... = GlobalTensor<float, GTShape_<...>, GTStride_<...>>;
GT_... v35 = GT_...(v32, v33, v34);
```

**转换过程**：

1. MemRef 的 shape/stride 被编码为 `pto::Shape<N1,N2,...>` 和 `pto::Stride<S1,S2,...>` 类型别名
2. 指针经过偏移计算（`base_ptr + offset`）
3. 构造 `GlobalTensor<float, Shape, Stride>` 对象
4. 偏移计算中的算术表达式由 EmitC 的 `arith` lowering 生成

---

### 8.3 TLOAD → TLOAD() 调用

**源码**：`PTOTLoadToTLOAD`（行 4063）

`.pto`：

```mlir
pto.tload ins(%13 : !pto.partition_tensor_view<32x32xf32>)
     outs(%4 : !pto.tile_buf<loc=mat, ...>)
```

**转换逻辑**（简化）：

```cpp
// 1. 获取 dst tile 变量
Value dst = peelUnrealized(adaptor.getDst());

// 2. 如果 src 是全局 MemRef，构造 GlobalTensor
Value srcArg = buildGlobalTensorFromMemref(rewriter, loc, src, ...);

// 3. 生成 emitc::CallOpaqueOp
rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TLOAD",
    ArrayAttr{}, ArrayAttr{}, ValueRange{dst, srcArg});
```

**C++ 输出**：

```cpp
TLOAD(v16, v35);   // TLOAD(tile, globalTensor)
```

**参数顺序**：`TLOAD(dst_tile, src_globalTensor)`——先目标后源。

---

### 8.4 set_flag / wait_flag → set_flag() / wait_flag() 调用

**源码**：`PTOSetFlagToEmitC`（行 4778）、`PTOWaitFlagToEmitC`（行 4805）

`.pto`：

```mlir
pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
```

**转换逻辑**：

```cpp
// extractSyncTokens() 将 PIPE/Event 属性转换为 C++ 标识符字符串
auto argsAttr = rewriter.getArrayAttr({
    emitc::OpaqueAttr::get(ctx, "PIPE_MTE2"),   // src_pipe
    emitc::OpaqueAttr::get(ctx, "PIPE_MTE1"),   // dst_pipe
    emitc::OpaqueAttr::get(ctx, "EVENT_ID0"),   // event_id
});
rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
    op, TypeRange{}, "set_flag", argsAttr, ArrayAttr{}, ValueRange{});
```

**C++ 输出**：

```cpp
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
```

PIPE/Event 枚举直接作为参数传入函数调用。无操作数（`ValueRange{}`），所有参数都是编译时常量。

---

### 8.5 TMOV → TMOV() 调用

**源码**：`PTOMovToEmitC`（行 8127）

`.pto`：

```mlir
pto.tmov ins(%4 : !pto.tile_buf<loc=mat, ...>)
    outs(%7 : !pto.tile_buf<loc=left, ...>)
```

**C++ 输出**：

```cpp
TMOV(v19, v16);   // TMOV(dst, src)
```

---

### 8.6 TMATMUL → TMATMUL() / TMATMUL_BIAS() / TMATMUL_ACC() 调用

**源码**：`PTOTMatmulToTMATMUL`（行 4284）

三种变体的转换逻辑相同：

`.pto`：

```mlir
// 无 bias
pto.tmatmul ins(%7, %8) outs(%9)

// 有 bias
pto.tmatmul.bias ins(%7, %8, %10) outs(%9)

// 累加
pto.tmatmul.acc ins(%9, %7, %8) outs(%9)
```

**转换逻辑**：

```cpp
// tmatmul → TMATMUL(dst, lhs, rhs)
rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TMATMUL",
    ArrayAttr{}, ArrayAttr{}, ValueRange{dst, lhs, rhs});

// tmatmul.bias → TMATMUL_BIAS(dst, lhs, rhs, bias)
// tmatmul.acc → TMATMUL_ACC(dst, dst, lhs, rhs)
```

**C++ 输出**：

```cpp
TMATMUL(v21, v19, v20);              // C = A × B
TMATMUL_BIAS(v21, v19, v20, v22);    // C = A × B + bias
TMATMUL_ACC(v21, v21, v19, v20);     // C += A × B
```

---

### 8.7 TSTORE → TSTORE() 调用

**源码**：`PTOTStoreToTSTORE`（行 4133）

`.pto`：

```mlir
pto.tstore ins(%9 : !pto.tile_buf<loc=acc, ...>)
     outs(%11 : !pto.partition_tensor_view<32x32xf32>)
```

**C++ 输出**：

```cpp
// GlobalTensor 构造（同 TLOAD 的逻辑）
__gm__ float* v64 = v1 + v63;
using GTShape_... = pto::Shape<1, 1, 1, 32, 32>;
...
GT_... v67 = GT_...(v64, v65, v66);
TSTORE(v67, v21);   // TSTORE(globalTensor, src_tile)
```

**注意参数顺序**：`TSTORE(dst_globalTensor, src_tile)`——与 TLOAD 相反的方向。

---

### 8.8 控制流转换

`.pto` 中的 `scf.for` 和 `scf.if` 由 MLIR 标准的 SCF→EmitC lowering 处理：

| .pto | C++ |
|---|---|
| `scf.for %arg5 = %c0 to %c8 step %c1` | `for (int32_t v23 = v12; v23 < v8; v23 += v11)` |
| `scf.if %arg4 { ... } else { ... }` | `if (v5) { ... } else { ... }` |
| `arith.cmpi eq, %arg5, %c0` | `bool v57 = v23 == v12` |

---

## 9. Pass 14：FormExpressions + CSE

`emitc::createFormExpressionsPass()` 将 EmitC 中的多步运算合并为单一表达式。CSE（Common Subexpression Elimination）消除重复计算。

---

## 10. C++ 发射与后处理

### 10.1 translateToCpp()

EmitC dialect 的标准翻译：将 `emitc::CallOpaqueOp` 翻译为 C 函数调用，`emitc::VarOp` 翻译为变量声明。

### 10.2 函数签名

`.pto`：

```mlir
func.func @RunTMATMULSplitK(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
  %arg2: !pto.ptr<f32>, %arg3: !pto.ptr<f32>, %arg4: i1)
```

**C++ 输出**：

```cpp
__global__ AICORE void RunTMATMULSplitK(
    __gm__ float* v1, __gm__ float* v2, __gm__ float* v3,
    __gm__ float* v4, bool v5)
```

映射：
- `!pto.ptr<f32>` → `__gm__ float*`（全局内存指针）
- `i1` → `bool`
- `__global__ AICORE` 来自函数属性（kernel 入口标记）

### 10.3 后处理重写（ptoas.cpp:1241–1247）

```cpp
rewriteTileGetSetValueMarkers(cppOutput);    // PTOAS__TILE_SET_VALUE → tile.SetValue
rewriteAsyncEventMarkers(cppOutput);         // 异步事件标记重写
rewritePtrScalarMarkers(cppOutput);          // 标量指针操作重写
rewriteEventIdArrayMarkers(cppOutput);       // event-id 数组下标重写
rewriteScalarConstantDecls(cppOutput);       // 标量常量声明优化
rewriteHoistedGlobalTensorDecls(cppOutput);  // GlobalTensor 提升优化
```

这些是对 EmitC 生成的 C++ 文本的后处理，修正 EmitC 无法直接表达的 Ascend 特有语法。

---

## 11. 完整映射表

| .pto 行 | .pto Op | C++ 行 | C++ 代码 | 转换 Pass |
|---|---|---|---|---|
| 1 | `module attributes` | 9–11 | `#include`, `__global__ AICORE void` | EmitC header + 函数属性 |
| 2 | `func.func @RunTMATMULSplitK` | 11 | `void RunTMATMULSplitK(...)` | EmitC func lowering |
| 3–12 | `arith.constant` | 12–21 | `unsigned/int32_t/int64_t vN = value` | 类型转换器 + EmitC |
| 17–23 | `pto.alloc_tile` | 23–36 | `Tile<...> vN; TASSIGN(vN, addr)` | PTOToEmitC + PlanMemory |
| 24 | `scf.for` | 37 | `for (int32_t v23 = ...)` | SCF→EmitC |
| 25 | `arith.muli` | 38 | `int32_t v24 = v23 * v10` | Arith→EmitC |
| 26–28 | `pto.partition_view` | 39–79 | 指针偏移 + GlobalTensor 构造 | ViewToMemref + PTOToEmitC |
| 29–30 | `pto.tload` | 80–81 | `TLOAD(v16, v35)` | PTOTLoadToTLOAD |
| 31–34 | `scf.if + pto.tload` | 82–85 | `if (v5) { TLOAD(...) }` | SCF→EmitC + PTOTLoadToTLOAD |
| 35 | `pto.set_flag` | 86 | `set_flag(PIPE_MTE2, PIPE_MTE1, ...)` | PTOSetFlagToEmitC |
| 36 | `pto.wait_flag` | 87 | `wait_flag(PIPE_MTE2, PIPE_MTE1, ...)` | PTOWaitFlagToEmitC |
| 37–38 | `pto.tmov` | 88–89 | `TMOV(v19, v16)`, `TMOV(v20, v17)` | PTOMovToEmitC |
| 39–42 | `scf.if + pto.tmov` | 90–93 | `if (v5) { TMOV(...) }` | SCF→EmitC + PTOMovToEmitC |
| 43 | `pto.set_flag` | 94 | `set_flag(PIPE_MTE1, PIPE_M, ...)` | PTOSetFlagToEmitC |
| 44 | `pto.wait_flag` | 95 | `wait_flag(PIPE_MTE1, PIPE_M, ...)` | PTOWaitFlagToEmitC |
| 45 | `arith.cmpi` | 96 | `bool v57 = v23 == v12` | Arith→EmitC |
| 46–54 | `scf.if + tmatmul variants` | 97–105 | `if/else { TMATMUL/TMATMUL_BIAS/TMATMUL_ACC }` | PTOTMatmulToTMATMUL |
| 55 | `pto.set_flag` | 106 | `set_flag(PIPE_M, PIPE_MTE2, ...)` | PTOSetFlagToEmitC |
| 56 | `pto.wait_flag` | 107 | `wait_flag(PIPE_M, PIPE_MTE2, ...)` | PTOWaitFlagToEmitC |
| 58 | `pto.set_flag` | 109 | `set_flag(PIPE_M, PIPE_FIX, ...)` | PTOSetFlagToEmitC |
| 59 | `pto.wait_flag` | 110 | `wait_flag(PIPE_M, PIPE_FIX, ...)` | PTOWaitFlagToEmitC |
| 60 | `pto.partition_view` | 111–123 | 指针偏移 + GlobalTensor 构造 | ViewToMemref + PTOToEmitC |
| 61 | `pto.tstore` | 124 | `TSTORE(v67, v21)` | PTOTStoreToTSTORE |
| 62 | `return` | 125 | `return` | EmitC |

---

## 12. PTO 地址空间 → C++ TileType 映射

定义在 `lib/PTO/Transforms/PTOToEmitC.cpp:260`（`tileRoleToken`）：

| .pto AddressSpace | C++ TileType | 硬件位置 |
|---|---|---|
| `mat` | `TileType::Mat` | 搬运缓冲区（L1） |
| `left` | `TileType::Left` | 左矩阵 L1 缓冲区 |
| `right` | `TileType::Right` | 右矩阵 L1 缓冲区 |
| `acc` | `TileType::Acc` | 累加器 L0C 缓冲区 |
| `bias` | `TileType::Bias` | 偏置缓冲区 |
| `vec` | `TileType::Vec` | 向量缓冲区 |
| `scaling` | `TileType::Scaling` | 缩放缓冲区 |

## 13. PTO 配置属性 → C++ 枚举映射

### BLayout（行 3617）

| .pto blayout | C++ 枚举 |
|---|---|
| `col_major` | `BLayout::ColMajor` |
| `row_major` | `BLayout::RowMajor` |

### SLayout（行 3626）

| .pto slayout | C++ 枚举 |
|---|---|
| `row_major` | `SLayout::RowMajor` |
| `col_major` | `SLayout::ColMajor` |
| `none_box` | `SLayout::NoneBox` |

### PadValue（行 3637）

| .pto pad | C++ 枚举 |
|---|---|
| `0` | `PadValue::Null` |
| `1` | `PadValue::Zero` |
| `2` | `PadValue::Max` |
| `3` | `PadValue::Min` |

---

## 14. 关键观察

1. **PTOToEmitC 是一个手动 lowering pass**（~11890 行），不使用标准的 DialectConversion 框架的自动推导，而是为每个 PTO op 手写转换规则。这是因为 PTO ops 语义与标准 MLIR ops 差异很大（硬件特定概念如 pipe、tile、address space）。

2. **EmitC 是桥梁**。PTO ops → EmitC ops → C++ 文本。EmitC 提供了 `CallOpaqueOp`（不透明函数调用）和 `OpaqueType`（不透明类型）来表达 pto-isa C++ 库的 API。

3. **GlobalTensor 构造是隐式的**。`.pto` 中的 `partition_view` + `tload` 组合在 C++ 中展开为完整的 GlobalTensor 构造（Shape/Stride 模板参数 + 指针偏移计算）。

4. **PlanMemory 的地址分配体现在 TASSIGN 宏中**。`TASSIGN(tile, offset)` 将 tile 绑定到本地内存地址，offset 由 PlanMemory pass 计算得出。

5. **后处理弥补 EmitC 的不足**。EmitC 无法直接表达 Ascend 特有的语法（如 `__gm__` 修饰符、tile.SetValue、event-id 数组下标等），这些通过字符串后处理解决。
