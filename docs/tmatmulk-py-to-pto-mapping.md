# tmatmulk.py → tmatmulk.pto 逐行映射分析

本文档详细追踪 `test/samples/MatMul/tmatmulk.py` 中每个 Python API 调用如何转换为 `test/samples/MatMul/tmatmulk.pto` 中的 PTO IR 文本。

核心转换链：

```
Python API (pto.py)
  → MLIR IR 构造 (mlir.ir Context + PTO 方言 ops)
    → module.operation.verify()
      → print(module)  →  .pto 文本输出
```

Python 调用直接构造 MLIR IR 对象，`print()` 时由 MLIR 的 assembly printer 输出文本格式。没有额外的代码生成步骤——Python API 就是 IR 构造器。

---

## 1. 方言注册

### Python（36–37 行）

```python
with Context() as ctx, Location.unknown():
    pto.register_dialect(ctx, load=True)
```

### PTO 输出

无直接对应输出。`register_dialect` 将 PTO 方言注册到 MLIR Context 中，使得后续可以使用 `pto.xxx` ops。但 `.pto` 文件的 `module` 隐式使用了该方言。

### 内部机制

`pto.register_dialect` 来自 C++ pybind11 原生扩展 `_pto`（`lib/Bindings/Python/PTOModule.cpp`），调用 `ctx.getOrLoadDialect<PTODialect>()`。

---

## 2. Module 创建与设备属性

### Python（39–40 行）

```python
module = builtin.ModuleOp()
module.attributes["pto.device-spec"] = StringAttr.get("Ascend910B1")
```

### PTO 输出（第 1 行）

```mlir
module attributes {"pto.device-spec" = "Ascend910B1"} {
```

### 映射

| Python | PTO | 说明 |
|---|---|---|
| `builtin.ModuleOp()` | `module { ... }` | 顶层 MLIR Module |
| `module.attributes["pto.device-spec"] = ...` | `attributes {"pto.device-spec" = "Ascend910B1"}` | 声明目标设备为 A3 |

---

## 3. 函数定义

### Python（137–140 行）

```python
fn_ty = func.FunctionType.get([ptr_out, ptr_a, ptr_b, ptr_bias, i1], [])
with InsertionPoint(module.body):
    fn = func.FuncOp("RunTMATMULSplitK", fn_ty)
    entry = fn.add_entry_block()
```

### PTO 输出（第 2 行）

```mlir
func.func @RunTMATMULSplitK(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
  %arg2: !pto.ptr<f32>, %arg3: !pto.ptr<f32>, %arg4: i1) {
```

### 映射

| Python | PTO |
|---|---|
| `FuncOp("RunTMATMULSplitK", fn_ty)` | `func.func @RunTMATMULSplitK(...)` |
| 参数 `[ptr_out, ptr_a, ptr_b, ptr_bias, i1]` | `%arg0: !pto.ptr<f32>, ..., %arg4: i1` |
| `entry.arguments` 解包 | `%arg0`...`%arg4` |

---

## 4. 常量定义

### Python（146–158 行）

```python
c0 = _idx_const(0)       # arith.ConstantOp(IndexType.get(), 0)
c1 = _idx_const(1)
cM = _idx_const(32)      # validM
cK = _idx_const(256)     # validK
cN = _idx_const(32)      # validN
cBASEK = _idx_const(32)
cIter = _idx_const(8)    # 256/32
cTileM = _idx_const(32)  # M
cTileN = _idx_const(32)  # N
```

### PTO 输出（第 3–12 行）

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c1_0 = arith.constant 1 : index       # cOne
%c32 = arith.constant 32 : index       # cM (= validM)
%c256 = arith.constant 256 : index     # cK (= validK)
%c32_1 = arith.constant 32 : index     # cN (= validN)
%c32_2 = arith.constant 32 : index     # cBASEK
%c8 = arith.constant 8 : index         # cIter
%c32_3 = arith.constant 32 : index     # cTileM
%c32_4 = arith.constant 32 : index     # cTileN
```

### 映射说明

- `_idx_const(v)` = `arith.ConstantOp(IndexType.get(), v).result`
- MLIR printer 自动对重复值添加后缀（`_0`, `_1`, `_2`, ...）以区分同名常量
- 值相同时（如多个 32），MLIR 为每个常量分配不同的 SSA 名称（`%c32`, `%c32_1`, ...），但它们是独立的 IR 节点

---

## 5. TensorView 创建

### Python（162–168 行）

```python
tvA  = pto.MakeTensorViewOp(tv2_a,  a_ptr,    [cM, cK],  [cK, c1]).result
tvB  = pto.MakeTensorViewOp(tv2_b,  b_ptr,    [cK, cN],  [cN, c1]).result
tvOut= pto.MakeTensorViewOp(tv2_out, out_ptr, [cM, cN],  [cN, c1]).result
tvBias=pto.MakeTensorViewOp(tv2_bias,bias_ptr,[cOne, cN],[cN, c1]).result
```

### PTO 输出（第 13–16 行）

```mlir
%0 = pto.make_tensor_view %arg1, shape = [%c32, %c256], strides = [%c256, %c1]
     : !pto.tensor_view<2xf32>
%1 = pto.make_tensor_view %arg2, shape = [%c256, %c32_1], strides = [%c32_1, %c1]
     : !pto.tensor_view<2xf32>
%2 = pto.make_tensor_view %arg0, shape = [%c32, %c32_1], strides = [%c32_1, %c1]
     : !pto.tensor_view<2xf32>
%3 = pto.make_tensor_view %arg3, shape = [%c1_0, %c32_1], strides = [%c32_1, %c1]
     : !pto.tensor_view<2xf32>
```

### 映射

| Python 参数 | PTO 语法位置 | 说明 |
|---|---|---|
| 第 1 个参数 `tv2_a` | `: !pto.tensor_view<2xf32>` | 结果类型（2 维 f32 视图） |
| 第 2 个参数 `a_ptr` | `%arg1` | 源指针操作数 |
| 第 3 个参数 `[cM, cK]` | `shape = [%c32, %c256]` | 形状属性 |
| 第 4 个参数 `[cK, c1]` | `strides = [%c256, %c1]` | 步长属性 |

注意：Python 传入参数的顺序（类型、指针、shape、stride）对应 ODS 中 `PTOMakeTensorViewOp` 的 operand + attribute 定义，MLIR printer 按 assembly format 输出。

---

## 6. Tile 分配

### Python（171–178 行）

```python
aMatTile   = pto.AllocTileOp(tile_buf_aMat).result     # MAT, 32x32, cfg_mat
bMatTile   = pto.AllocTileOp(tile_buf_bMat).result     # MAT, 32x32, cfg_mat
biasDataTile=pto.AllocTileOp(tile_buf_biasData).result  # MAT, 1x32, cfg_mat_bias
aTile      = pto.AllocTileOp(tile_buf_aTile).result     # LEFT, 32x32, cfg_left
bTile      = pto.AllocTileOp(tile_buf_bTile).result     # RIGHT, 32x32, cfg_right
cTile      = pto.AllocTileOp(tile_buf_cTile).result     # ACC, 32x32, cfg_acc
biasTile   = pto.AllocTileOp(tile_buf_biasTile).result  # BIAS, 1x32, cfg_bias
```

### PTO 输出（第 17–23 行）

```mlir
%4 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32,
     v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=512, pad=0>
%5 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32,
     v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=512, pad=0>
%6 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=1, cols=32,
     v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%7 = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f32, rows=32, cols=32,
     v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=512, pad=0>
%8 = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f32, rows=32, cols=32,
     v_row=32, v_col=32, blayout=row_major, slayout=col_major, fractal=512, pad=0>
%9 = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
     v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
%10 = pto.alloc_tile : !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=32,
     v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

### 映射

`TileBufType.get()` 的参数完全编码到 PTO 类型打印中：

```
Python: TileBufType.get([M, BASEK], t_a, left, [M, BASEK], cfg_left)
              ↓           ↓        ↓      ↓         ↓          ↓
PTO:   tile_buf<loc=left, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=col_major,
              slayout=row_major, fractal=512, pad=0>
```

| Python 参数 | PTO 字段 |
|---|---|
| `[M, BASEK]` = `[32, 32]` | `rows=32, cols=32` |
| `t_a` = `F32Type` | `dtype=f32` |
| `left` = `AddressSpace.LEFT` | `loc=left` |
| `[M, BASEK]` = `[32, 32]` | `v_row=32, v_col=32` |
| `cfg_left` | `blayout=col_major, slayout=row_major, fractal=512, pad=0` |

---

## 7. Split-K 循环

### Python（184 行）

```python
loop = scf.ForOp(c0, cIter, c1, [])
```

### PTO 输出（第 24 行）

```mlir
scf.for %arg5 = %c0 to %c8 step %c1 {
```

### 映射

| Python | PTO |
|---|---|
| `c0` (lower bound) | `%c0` |
| `cIter` = `_idx_const(8)` (upper bound) | `%c8` |
| `c1` (step) | `%c1` |
| `[]` (no loop-carried values) | 无 `iter_args` |
| `loop.induction_variable` → `i` | `%arg5` |

---

## 8. K 偏移计算

### Python（189 行）

```python
kOff = arith.MulIOp(i, cBASEK).result
```

### PTO 输出（第 25 行）

```mlir
%12 = arith.muli %arg5, %c32_2 : index
```

---

## 9. PartitionView 切分

### Python（192–194 行）

```python
svA   = pto.PartitionViewOp(tile_view_a, tvA,  offsets=[c0, kOff], sizes=[cTileM, cBASEK]).result
svB   = pto.PartitionViewOp(tile_view_b, tvB,  offsets=[kOff, c0], sizes=[cBASEK, cTileN]).result
svBias= pto.PartitionViewOp(tile_view_bias,tvBias,offsets=[c0, c0],sizes=[cOne, cTileN]).result
```

### PTO 输出（第 26–28 行）

```mlir
%13 = pto.partition_view %0, offsets = [%c0, %12], sizes = [%c32_3, %c32_2]
     : !pto.tensor_view<2xf32> -> !pto.partition_tensor_view<32x32xf32>
%14 = pto.partition_view %1, offsets = [%12, %c0], sizes = [%c32_2, %c32_4]
     : !pto.tensor_view<2xf32> -> !pto.partition_tensor_view<32x32xf32>
%15 = pto.partition_view %3, offsets = [%c0, %c0], sizes = [%c1_0, %c32_4]
     : !pto.tensor_view<2xf32> -> !pto.partition_tensor_view<1x32xf32>
```

### 映射

| Python 参数 | PTO 语法 | 说明 |
|---|---|---|
| `tile_view_a`（结果类型） | `-> !pto.partition_tensor_view<32x32xf32>` | 切片结果的类型 |
| `tvA`（源视图） | `%0` | 被切分的 TensorView |
| `offsets=[c0, kOff]` | `offsets = [%c0, %12]` | 切片起始位置 |
| `sizes=[cTileM, cBASEK]` | `sizes = [%c32_3, %c32_2]` | 切片大小 |

---

## 10. TLOAD 操作

### Python（198–199 行）

```python
pto.TLoadOp(None, svA, aMatTile)
pto.TLoadOp(None, svB, bMatTile)
```

### PTO 输出（第 29–30 行）

```mlir
pto.tload ins(%13 : !pto.partition_tensor_view<32x32xf32>)
     outs(%4 : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, ...>)
pto.tload ins(%14 : !pto.partition_tensor_view<32x32xf32>)
     outs(%5 : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, ...>)
```

### 映射

| Python 参数位置 | PTO 关键字 | 说明 |
|---|---|---|
| `None`（第 1 个参数） | 无 | 空 output（ODS 中定义为可选） |
| `svA`（第 2 个参数） | `ins(%13 : ...)` | 输入 PartitionTensorView |
| `aMatTile`（第 3 个参数） | `outs(%4 : ...)` | 目标 Tile 缓冲区 |

### 条件 Bias TLOAD（201–206 行）

```python
if_load_bias = scf.IfOp(isBias, [], hasElse=True)
with InsertionPoint(if_load_bias.then_block):
    pto.TLoadOp(None, svBias, biasDataTile)
    scf.YieldOp([])
with InsertionPoint(if_load_bias.else_block):
    scf.YieldOp([])
```

PTO 输出（第 31–34 行）：

```mlir
scf.if %arg4 {
  pto.tload ins(%15 : !pto.partition_tensor_view<1x32xf32>)
       outs(%6 : !pto.tile_buf<loc=mat, dtype=f32, rows=1, cols=32, ...>)
} else {
}
```

`isBias`（`%arg4 : i1`）作为 `scf.if` 的条件，只有当 `isBias=true` 时才加载 bias。

---

## 11. 同步：record_event / wait_event

这是最关键的映射——涉及 **两层转换**。

### Python（209–210 行）

```python
pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
pto.wait_event  (TLOAD, TMOV_M2L, EVENT_ID0)
```

### 中间层：Python → MLIR IR 构造

`pto.py` 中的转换过程：

```python
# record_event 内部实现 (python/pto/dialects/pto.py:269)
def record_event(src_op, dst_op, event_id, ...):
    ctx = _ods_ir.Context.current
    return _pto_ops_gen.record_event(
        _ensure_sync_attr(src_op, ctx),    # SyncOpType.TLOAD → SyncOpTypeAttr
        _ensure_sync_attr(dst_op, ctx),    # SyncOpType.TMOV_M2L → SyncOpTypeAttr
        _ensure_event_attr(event_id, ctx), # EVENT_ID0 → EventAttr
    )
```

`_ensure_sync_attr` 将 Python 枚举值转换为 MLIR 属性：
- `TLOAD` = `SyncOpType.TLOAD` → `SyncOpTypeAttr.get(SyncOpType::TLOAD)`
- `TMOV_M2L` = `SyncOpType.TMOV_M2L` → `SyncOpTypeAttr.get(SyncOpType::TMOV_M2L)`
- `EVENT_ID0` = `EVENT.EVENT_ID0` → `EventAttr.get(EVENT::EVENT_ID0)`

这构造了高层 `record_event` / `wait_event` ops（不是 `set_flag`/`wait_flag`）。

### 但是 .pto 中已经显示为 set_flag / wait_flag

查看 `.pto` 输出（第 35–36 行）：

```mlir
pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
```

**这意味着 `tmatmulk.pto` 不是直接由 Python `print(module)` 生成的！** 它经过了 `ptoas` 的 pass pipeline 中的 `LoweringSyncToPipe` pass。

### 完整转换链

```
Python:
  pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
    ↓ (pto.py 封装)
MLIR IR:
  pto.record_event [<TLOAD>, <TMOV_M2L>, <EVENT_ID0>]
    ↓ (LoweringSyncToPipe pass)
    ↓ mapSyncOpTypeToPipe(TLOAD) → PIPE_MTE2
    ↓ mapSyncOpTypeToPipe(TMOV_M2L) → PIPE_MTE1
MLIR IR (lowered):
  pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    ↓ (print)
.pto 输出:
  pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
```

### SyncOpType → PIPE 映射表

定义在 `lib/PTO/IR/PTOSyncUtils.cpp:25`：

| SyncOpType | PIPE | 含义 |
|---|---|---|
| `TLOAD` | `PIPE_MTE2` | 数据搬运（GM → MAT） |
| `TSTORE_VEC` | `PIPE_MTE3` | 向量存储 |
| `TSTORE_ACC` | `PIPE_FIX` | 累加器存储 |
| `TMOV_M2L` / `TMOV_M2B` | `PIPE_MTE1` | MAT → LEFT/BIAS 搬运 |
| `TMOV_M2S` | `PIPE_FIX` | MAT → SCALAR 搬运 |
| `TMOV_M2V` | `PIPE_V` | MAT → VEC 搬运 |
| `TMOV_V2M` | `PIPE_FIX` | VEC → MAT 搬运 |
| `TMATMUL` | `PIPE_M` | Cube 矩阵乘法 |
| `TVEC` / `TVECWAIT_EVENT` | `PIPE_V` | 向量计算 |

### 脚本中三组同步的完整映射

| Python | LoweringSyncToPipe | .pto |
|---|---|---|
| `record_event(TLOAD, TMOV_M2L, EVENT_ID0)` | `set_flag[MTE2, MTE1, EVENT_ID0]` | 第 35 行 |
| `wait_event(TLOAD, TMOV_M2L, EVENT_ID0)` | `wait_flag[MTE2, MTE1, EVENT_ID0]` | 第 36 行 |
| `record_event(TMOV_M2L, TMATMUL, EVENT_ID0)` | `set_flag[MTE1, M, EVENT_ID0]` | 第 43 行 |
| `wait_event(TMOV_M2L, TMATMUL, EVENT_ID0)` | `wait_flag[MTE1, M, EVENT_ID0]` | 第 44 行 |
| `record_event(TMATMUL, TLOAD, EVENT_ID0)` | `set_flag[M, MTE2, EVENT_ID0]` | 第 55 行 |
| `wait_event(TMATMUL, TLOAD, EVENT_ID0)` | `wait_flag[M, MTE2, EVENT_ID0]` | 第 56 行 |
| `record_event(TMATMUL, TSTORE_ACC, EVENT_ID0)` | `set_flag[M, FIX, EVENT_ID0]` | 第 58 行 |
| `wait_event(TMATMUL, TSTORE_ACC, EVENT_ID0)` | `wait_flag[M, FIX, EVENT_ID0]` | 第 59 行 |

---

## 12. TMOV 操作

### Python（214–215 行）

```python
pto.TMovOp(None, aMatTile, aTile)    # MAT → LEFT
pto.TMovOp(None, bMatTile, bTile)    # MAT → RIGHT
```

### PTO 输出（第 37–38 行）

```mlir
pto.tmov ins(%4 : !pto.tile_buf<loc=mat, ...>)
    outs(%7 : !pto.tile_buf<loc=left, ...>)
pto.tmov ins(%5 : !pto.tile_buf<loc=mat, ...>)
    outs(%8 : !pto.tile_buf<loc=right, ...>)
```

### 条件 Bias TMOV（217–222 行）

```python
if_mov_bias = scf.IfOp(isBias, [], hasElse=True)
with InsertionPoint(if_mov_bias.then_block):
    pto.TMovOp(None, biasDataTile, biasTile)  # MAT → BIAS
    scf.YieldOp([])
```

PTO 输出（第 39–42 行）：

```mlir
scf.if %arg4 {
  pto.tmov ins(%6 : !pto.tile_buf<loc=mat, ...>)
      outs(%10 : !pto.tile_buf<loc=bias, ...>)
} else {
}
```

---

## 13. TMATMUL 操作

### Python（228–250 行）

```python
is_i0 = arith.CmpIOp(CmpIPredicate.eq, i, c0).result
if_i0 = scf.IfOp(is_i0, [], hasElse=True)

with InsertionPoint(if_i0.then_block):
    if_bias0 = scf.IfOp(isBias, [], hasElse=True)
    with InsertionPoint(if_bias0.then_block):
        pto.TMatmulBiasOp(None, aTile, bTile, biasTile, cTile)  # i==0 && bias
    with InsertionPoint(if_bias0.else_block):
        pto.TMatmulOp(None, aTile, bTile, cTile)                 # i==0 && !bias

with InsertionPoint(if_i0.else_block):
    pto.TMatmulAccOp(None, cTile, aTile, bTile, cTile)          # i!=0
```

### PTO 输出（第 45–54 行）

```mlir
%16 = arith.cmpi eq, %arg5, %c0 : index
scf.if %16 {
  scf.if %arg4 {
    pto.tmatmul.bias ins(%7, %8, %10 : <left...>, <right...>, <bias...>)
        outs(%9 : <acc...>)
  } else {
    pto.tmatmul ins(%7, %8 : <left...>, <right...>)
        outs(%9 : <acc...>)
  }
} else {
  pto.tmatmul.acc ins(%9, %7, %8 : <acc...>, <left...>, <right...>)
      outs(%9 : <acc...>)
}
```

### 映射

| Python Op | PTO Op | 输入操作数 | 输出操作数 |
|---|---|---|---|
| `TMatmulBiasOp(None, aTile, bTile, biasTile, cTile)` | `tmatmul.bias` | `ins(left, right, bias)` | `outs(acc)` |
| `TMatmulOp(None, aTile, bTile, cTile)` | `tmatmul` | `ins(left, right)` | `outs(acc)` |
| `TMatmulAccOp(None, cTile, aTile, bTile, cTile)` | `tmatmul.acc` | `ins(acc, left, right)` | `outs(acc)` |

注意 `tmatmul.acc` 的第一个输入是 `cTile`（ACC），即当前累加值，传入传出都是 `%9`。

---

## 14. 循环后 TSTORE

### Python（258–265 行）

```python
pto.record_event(TMATMUL, TSTORE_ACC, EVENT_ID0)
pto.wait_event  (TMATMUL, TSTORE_ACC, EVENT_ID0)

svOut = pto.PartitionViewOp(tile_view_out, tvOut, offsets=[c0, c0], sizes=[cTileM, cTileN]).result
pto.TStoreOp(None, cTile, svOut)
```

### PTO 输出（第 58–61 行）

```mlir
pto.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
pto.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
%11 = pto.partition_view %2, offsets = [%c0, %c0], sizes = [%c32_3, %c32_4]
     : !pto.tensor_view<2xf32> -> !pto.partition_tensor_view<32x32xf32>
pto.tstore ins(%9 : !pto.tile_buf<loc=acc, ...>)
     outs(%11 : !pto.partition_tensor_view<32x32xf32>)
```

### 映射

| Python | 经过 Lowering | PTO |
|---|---|---|
| `record_event(TMATMUL, TSTORE_ACC, EVENT_ID0)` | `TMATMUL→PIPE_M, TSTORE_ACC→PIPE_FIX` | `set_flag[M, FIX, EVENT_ID0]` |
| `wait_event(TMATMUL, TSTORE_ACC, EVENT_ID0)` | 同上 | `wait_flag[M, FIX, EVENT_ID0]` |

---

## 15. 函数返回

### Python（267–269 行）

```python
func.ReturnOp([])
module.operation.verify()
return module
```

### PTO 输出（第 62–64 行）

```mlir
    return
  }
}
```

`verify()` 验证 IR 语义正确性（类型匹配、操作数数量等），不产生输出。`print(module)` 输出完整 `.pto`。

---

## 总结：Python → PTO 转换的两种机制

### 机制 1：直接映射（大多数 ops）

Python API 直接构造 MLIR IR，`print()` 时由 assembly printer 输出：

```
pto.AllocTileOp(tile_buf_type)  →  pto.alloc_tile : !pto.tile_buf<...>
pto.TLoadOp(None, src, dst)     →  pto.tload ins(src) outs(dst)
pto.TMovOp(None, src, dst)      →  pto.tmov ins(src) outs(dst)
pto.TMatmulOp(None, a, b, c)    →  pto.tmatmul ins(a, b) outs(c)
```

### 机制 2：经过 Pass 转换（同步 ops）

高层同步 API 先构造 `record_event`/`wait_event` ops，再由 `LoweringSyncToPipe` pass 转换为低层 `set_flag`/`wait_flag`：

```
pto.record_event(SyncOpType, SyncOpType, Event)
  → pto.record_event [<TLOAD>, <TMOV_M2L>, <EVENT_ID0>]
    → [LoweringSyncToPipe pass]
      → pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
```

这意味着 `tmatmulk.pto` 文件是经过 `python3 tmatmulk.py | ptoas --emit-pto-ir` 处理后的结果，而不是纯 Python `print()` 的直接输出。

### 全量映射表

| Python 行号 | Python 调用 | PTO 行号 | PTO Op |
|---|---|---|---|
| 40 | `module.attributes[...]` | 1 | `module attributes {...}` |
| 139 | `FuncOp("RunTMATMULSplitK", ...)` | 2 | `func.func @RunTMATMULSplitK` |
| 146–158 | `_idx_const(...)` | 3–12 | `arith.constant ... : index` |
| 162 | `MakeTensorViewOp(tv2_a, a_ptr, ...)` | 13 | `pto.make_tensor_view %arg1, ...` |
| 164 | `MakeTensorViewOp(tv2_b, b_ptr, ...)` | 14 | `pto.make_tensor_view %arg2, ...` |
| 166 | `MakeTensorViewOp(tv2_out, out_ptr, ...)` | 15 | `pto.make_tensor_view %arg0, ...` |
| 168 | `MakeTensorViewOp(tv2_bias, bias_ptr, ...)` | 16 | `pto.make_tensor_view %arg3, ...` |
| 171 | `AllocTileOp(tile_buf_aMat)` | 17 | `pto.alloc_tile : !pto.tile_buf<loc=mat, ...>` |
| 172 | `AllocTileOp(tile_buf_bMat)` | 18 | `pto.alloc_tile : !pto.tile_buf<loc=mat, ...>` |
| 173 | `AllocTileOp(tile_buf_biasData)` | 19 | `pto.alloc_tile : !pto.tile_buf<loc=mat, ...>` |
| 175 | `AllocTileOp(tile_buf_aTile)` | 20 | `pto.alloc_tile : !pto.tile_buf<loc=left, ...>` |
| 176 | `AllocTileOp(tile_buf_bTile)` | 21 | `pto.alloc_tile : !pto.tile_buf<loc=right, ...>` |
| 177 | `AllocTileOp(tile_buf_cTile)` | 22 | `pto.alloc_tile : !pto.tile_buf<loc=acc, ...>` |
| 178 | `AllocTileOp(tile_buf_biasTile)` | 23 | `pto.alloc_tile : !pto.tile_buf<loc=bias, ...>` |
| 184 | `scf.ForOp(c0, cIter, c1, [])` | 24 | `scf.for %arg5 = %c0 to %c8 step %c1` |
| 189 | `arith.MulIOp(i, cBASEK)` | 25 | `arith.muli %arg5, %c32_2` |
| 192 | `PartitionViewOp(tile_view_a, tvA, ...)` | 26 | `pto.partition_view %0, ...` |
| 193 | `PartitionViewOp(tile_view_b, tvB, ...)` | 27 | `pto.partition_view %1, ...` |
| 194 | `PartitionViewOp(tile_view_bias, ...)` | 28 | `pto.partition_view %3, ...` |
| 198 | `TLoadOp(None, svA, aMatTile)` | 29 | `pto.tload ins(%13) outs(%4)` |
| 199 | `TLoadOp(None, svB, bMatTile)` | 30 | `pto.tload ins(%14) outs(%5)` |
| 201–206 | `scf.IfOp(isBias) + TLoadOp(bias)` | 31–34 | `scf.if %arg4 { pto.tload ... }` |
| 209 | `record_event(TLOAD, TMOV_M2L, ...)` | 35 | `pto.set_flag[<PIPE_MTE2>, ...]` |
| 210 | `wait_event(TLOAD, TMOV_M2L, ...)` | 36 | `pto.wait_flag[<PIPE_MTE2>, ...]` |
| 214 | `TMovOp(None, aMatTile, aTile)` | 37 | `pto.tmov ins(%4) outs(%7)` |
| 215 | `TMovOp(None, bMatTile, bTile)` | 38 | `pto.tmov ins(%5) outs(%8)` |
| 217–222 | `scf.IfOp(isBias) + TMovOp(bias)` | 39–42 | `scf.if %arg4 { pto.tmov ... }` |
| 225 | `record_event(TMOV_M2L, TMATMUL, ...)` | 43 | `pto.set_flag[<PIPE_MTE1>, ...]` |
| 226 | `wait_event(TMOV_M2L, TMATMUL, ...)` | 44 | `pto.wait_flag[<PIPE_MTE1>, ...]` |
| 229 | `arith.CmpIOp(eq, i, c0)` | 45 | `arith.cmpi eq, %arg5, %c0` |
| 238 | `TMatmulBiasOp(None, a, b, bias, c)` | 48 | `pto.tmatmul.bias ins(%7,%8,%10) outs(%9)` |
| 242 | `TMatmulOp(None, a, b, c)` | 50 | `pto.tmatmul ins(%7,%8) outs(%9)` |
| 249 | `TMatmulAccOp(None, c, a, b, c)` | 53 | `pto.tmatmul.acc ins(%9,%7,%8) outs(%9)` |
| 253 | `record_event(TMATMUL, TLOAD, ...)` | 55 | `pto.set_flag[<PIPE_M>, <PIPE_MTE2>, ...]` |
| 254 | `wait_event(TMATMUL, TLOAD, ...)` | 56 | `pto.wait_flag[<PIPE_M>, <PIPE_MTE2>, ...]` |
| 259 | `record_event(TMATMUL, TSTORE_ACC, ...)` | 58 | `pto.set_flag[<PIPE_M>, <PIPE_FIX>, ...]` |
| 260 | `wait_event(TMATMUL, TSTORE_ACC, ...)` | 59 | `pto.wait_flag[<PIPE_M>, <PIPE_FIX>, ...]` |
| 264 | `PartitionViewOp(tile_view_out, tvOut, ...)` | 60 | `pto.partition_view %2, ...` |
| 265 | `TStoreOp(None, cTile, svOut)` | 61 | `pto.tstore ins(%9) outs(%11)` |
| 267 | `func.ReturnOp([])` | 62 | `return` |
