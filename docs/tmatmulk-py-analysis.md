# tmatmulk.py 源码解析

本文档对 `test/samples/MatMul/tmatmulk.py` 进行逐段解析，说明其如何通过 Python PTO 绑定构建 Split-K 矩阵乘法的 PTO IR。

---

## 功能概要

构建 Split-K MatMul 的 PTO IR：`C = A×B + bias`，沿 K 维度切分为 `K/BASEK` 次迭代，每次执行 `TLOAD → TMOV → TMATMUL` 流水线。

默认参数：`M=32, K=256, N=32, BASEK=32`，即 8 次迭代。

---

## 函数签名

```python
build(M=32, K=256, N=32, validM=32, validK=256, validN=32,
      BASEK=32, s_fractal_ab=512, s_fractal_c=1024)
```

| 参数 | 含义 |
|---|---|
| `M, K, N` | 矩阵维度，C = A[M,K] × B[K,N] + bias[1,N] |
| `validM, validK, validN` | 实际有效数据维度（用于 padding 场景） |
| `BASEK` | 每次迭代的 K 切分大小（256/32 = 8 次迭代） |
| `s_fractal_ab` | A/B 矩阵的分形缓冲区大小 |
| `s_fractal_c` | C 矩阵（累加器）的分形缓冲区大小 |

---

## 阶段 1：方言注册与 Module 创建（36–40 行）

```python
pto.register_dialect(ctx, load=True)
module = builtin.ModuleOp()
module.attributes["pto.device-spec"] = StringAttr.get("Ascend910B1")
```

- 注册 PTO 方言到 MLIR Context
- 创建顶层 ModuleOp
- 声明目标设备为 A3（`Ascend910B1`）

---

## 阶段 2：类型定义（42–66 行）

三层类型体系：

| 层级 | Python 构造 | 生成的 PTO 类型 | 用途 |
|---|---|---|---|
| 元素类型 | `F32Type.get()` | `f32` | 数据精度 |
| 指针类型 | `pto.PtrType.get(f32)` | `!pto.ptr<f32>` | GM（全局内存）数据指针 |
| TensorView | `pto.TensorViewType.get(2, f32)` | `!pto.tensor_view<2xf32>` | 对 GM 数据的二维视图（shape + stride） |
| PartitionTensorView | `pto.PartitionTensorViewType.get([32,32], f32)` | `!pto.partition_tensor_view<32x32xf32>` | 切分后的局部视图 |

TensorView 定义了 GM 数据的逻辑形状和步长：

```python
# A: [32, 256] 行主序，stride=[256, 1]
tvA = pto.MakeTensorViewOp(tv2_a, a_ptr, [cM, cK], [cK, c1])
```

---

## 阶段 3：地址空间与 Tile 配置（68–133 行）

### 地址空间

5 个地址空间对应 Ascend AICORE 的物理存储：

```
┌─────────────────────────────────────────┐
│  GM (Global Memory)                     │
│  全局内存，存放输入/输出张量              │
├─────────────────────────────────────────┤
│  MAT                                    │
│  搬运缓冲区，TLOAD 从 GM 搬数据到此处    │
├─────────────────────────────────────────┤
│  LEFT / RIGHT / BIAS                    │
│  L1 缓冲区，TMOV 从 MAT 搬数据到此处     │
├─────────────────────────────────────────┤
│  ACC                                    │
│  L0C 累加缓冲区，Cube 引擎的计算结果     │
└─────────────────────────────────────────┘
```

### Tile 配置

每个 Tile 配置包含 4 个属性：

| 属性 | 含义 | 可选值 |
|---|---|---|
| `BLayout` | 块布局 | `ColMajor`, `RowMajor` |
| `SLayout` | 存储布局 | `RowMajor`, `ColMajor`, `NoneBox` |
| `fractal` | 分形大小 | 512 (AB), 1024 (C) |
| `pad` | 填充策略 | `Null` |

6 种 Tile 配置：

| 配置名 | 地址空间 | BLayout | SLayout | fractal | 用途 |
|---|---|---|---|---|---|
| `cfg_mat` | MAT | ColMajor | RowMajor | 512 | A/B 搬运缓冲区 |
| `cfg_mat_bias` | MAT | RowMajor | NoneBox | 512 | Bias 搬运缓冲区 |
| `cfg_left` | LEFT | ColMajor | RowMajor | 512 | A 计算缓冲区 |
| `cfg_right` | RIGHT | RowMajor | ColMajor | 512 | B 计算缓冲区 |
| `cfg_acc` | ACC | ColMajor | RowMajor | 1024 | C 累加缓冲区 |
| `cfg_bias` | BIAS | RowMajor | NoneBox | 512 | Bias 计算缓冲区 |

### TileBuf 类型

TileBuf 类型编码了完整的硬件 tile 信息：

```python
tile_buf_aTile = pto.TileBufType.get([M, BASEK], t_a, left, [M, BASEK], cfg_left)
# → !pto.tile_buf<loc=left, dtype=f32, rows=32, cols=32,
#                 v_row=32, v_col=32, blayout=col_major,
#                 slayout=row_major, fractal=512, pad=0>
```

参数依次为：[形状], 元素类型, 地址空间, [有效形状], 配置属性。

---

## 阶段 4：函数定义与 TensorView 构建（136–168 行）

### 函数签名

```
func.func @RunTMATMULSplitK(
    %out_ptr: !pto.ptr<f32>,     // 输出矩阵 C 的 GM 指针
    %a_ptr:   !pto.ptr<f32>,     // 输入矩阵 A 的 GM 指针
    %b_ptr:   !pto.ptr<f32>,     // 输入矩阵 B 的 GM 指针
    %bias_ptr:!pto.ptr<f32>,     // 偏置向量的 GM 指针
    %isBias:  i1                 // 是否使用 bias（运行时控制）
)
```

### TensorView 构建

将 GM 指针映射为二维视图：

```python
# A: [32, 256], stride = [256, 1]
tvA = pto.MakeTensorViewOp(tv2_a, a_ptr, [cM, cK], [cK, c1])
# B: [256, 32], stride = [32, 1]
tvB = pto.MakeTensorViewOp(tv2_b, b_ptr, [cK, cN], [cN, c1])
# OUT: [32, 32], stride = [32, 1]
tvOut = pto.MakeTensorViewOp(tv2_out, out_ptr, [cM, cN], [cN, c1])
# BIAS: [1, 32], stride = [32, 1]
tvBias = pto.MakeTensorViewOp(tv2_bias, bias_ptr, [cOne, cN], [cN, c1])
```

---

## 阶段 5：Tile 分配（170–178 行）

分配 7 个硬件 Tile 缓冲区：

| 变量名 | 地址空间 | 大小 | 配置 | 用途 |
|---|---|---|---|---|
| `aMatTile` | MAT | 32×32 | cfg_mat | A 的搬运缓冲区 |
| `bMatTile` | MAT | 32×32 | cfg_mat | B 的搬运缓冲区 |
| `biasDataTile` | MAT | 1×32 | cfg_mat_bias | Bias 的搬运缓冲区 |
| `aTile` | LEFT | 32×32 | cfg_left | A 的计算缓冲区 |
| `bTile` | RIGHT | 32×32 | cfg_right | B 的计算缓冲区 |
| `cTile` | ACC | 32×32 | cfg_acc | C 的累加缓冲区 |
| `biasTile` | BIAS | 1×32 | cfg_bias | Bias 的计算缓冲区 |

---

## 阶段 6：Split-K 循环体（184–256 行）

核心计算循环，迭代 `K/BASEK = 8` 次。

### 循环结构

```python
loop = scf.ForOp(c0, cIter, c1, [])   # for i = 0 to 8 step 1
```

### 单次迭代流程

```
  ┌─────────────────────────────────────────────────────┐
  │  1. 计算 K 偏移: kOff = i * BASEK                    │
  │  2. 切分视图: partition_view 取出当前 K 切片           │
  │                                                      │
  │  3. TLOAD (GM → MAT)             ← PIPE_MTE2         │
  │     tload A[i:i+1, kOff:kOff+BASEK] → aMatTile       │
  │     tload B[kOff:kOff+BASEK, :] → bMatTile           │
  │     if isBias: tload bias → biasDataTile             │
  │                                                      │
  │  4. 同步: MTE2 → MTE1 (set_flag + wait_flag)         │
  │                                                      │
  │  5. TMOV (MAT → LEFT/RIGHT/BIAS) ← PIPE_MTE1        │
  │     tmov aMatTile → aTile                            │
  │     tmov bMatTile → bTile                            │
  │     if isBias: tmov biasDataTile → biasTile          │
  │                                                      │
  │  6. 同步: MTE1 → M (set_flag + wait_flag)            │
  │                                                      │
  │  7. TMATMUL                       ← PIPE_M (Cube)    │
  │     if i==0 && isBias:  tmatmul.bias  (清零+bias)    │
  │     elif i==0:          tmatmul       (清零)         │
  │     else:               tmatmul.acc   (累加)         │
  │                                                      │
  │  8. 同步: M → MTE2 (set_flag + wait_flag)            │
  └─────────────────────────────────────────────────────┘
```

### PartitionView 切分

每次迭代从完整 TensorView 中切出当前 K 切片：

```python
# A 的切片：行不变，列取 [kOff : kOff+BASEK]
svA = pto.PartitionViewOp(tile_view_a, tvA, offsets=[c0, kOff], sizes=[cTileM, cBASEK])

# B 的切片：行取 [kOff : kOff+BASEK]，列不变
svB = pto.PartitionViewOp(tile_view_b, tvB, offsets=[kOff, c0], sizes=[cBASEK, cTileN])
```

### 同步机制

```python
pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
# → set_flag[from_pipe=MTE2, to_pipe=MTE1, event_id=0]
# 含义：MTE2 管道完成 TLOAD 后，通知 MTE1 管道

pto.wait_event(TLOAD, TMOV_M2L, EVENT_ID0)
# → wait_flag[from_pipe=MTE2, to_pipe=MTE1, event_id=0]
# 含义：MTE1 管道等待 MTE2 完成
```

三重同步对应三对相邻管道：

| 同步点 | from → to | 语义 |
|---|---|---|
| TLOAD 完成后 | MTE2 → MTE1 | TLOAD 数据就绪，可以开始 TMOV |
| TMOV 完成后 | MTE1 → M | TMOV 数据就绪，可以开始 TMATMUL |
| TMATMUL 完成后 | M → MTE2 | 计算完成，可以开始下一次迭代的 TLOAD |

### TMATMUL 三种模式

| Python Op | PTO Op | 用途 | 行为 |
|---|---|---|---|
| `pto.TMatmulOp()` | `tmatmul` | 首次迭代（无 bias） | L0C 清零，C = A×B |
| `pto.TMatmulBiasOp()` | `tmatmul.bias` | 首次迭代（有 bias） | L0C 清零，C = A×B + bias |
| `pto.TMatmulAccOp()` | `tmatmul.acc` | 后续迭代 | L0C 保留，C += A×B |

选择逻辑：

```python
if i == 0:
    if isBias:
        tmatmul.bias(A, B, bias, C)   # 首次 + bias
    else:
        tmatmul(A, B, C)              # 首次无 bias
else:
    tmatmul.acc(C, A, B, C)           # 累加
```

---

## 阶段 7：循环后写回（258–267 行）

```python
# 同步：等待最后一个 TMATMUL 完成
pto.record_event(TMATMUL, TSTORE_ACC, EVENT_ID0)
pto.wait_event(TMATMUL, TSTORE_ACC, EVENT_ID0)

# TSTORE：将 ACC tile 写回 GM
svOut = pto.PartitionViewOp(tile_view_out, tvOut, offsets=[c0, c0], sizes=[cTileM, cTileN])
pto.TStoreOp(None, cTile, svOut)

func.ReturnOp([])
```

`record_event(TMATMUL, TSTORE_ACC, ...)` 在 `.pto` 输出中会被 LowerFrontendPipeOps pass 转换为 `set_flag[PIPE_M, PIPE_FIX, ...]`。

---

## 数据流总图

```
  GM（全局内存）
  A[M,K]  B[K,N]  bias[1,N]
    │       │       │
    │ TLOAD (PIPE_MTE2)
    ▼       ▼       ▼
  aMat   bMat   biasData        ← MAT 地址空间
    │       │       │
    │ TMOV (PIPE_MTE1)
    ▼       ▼       ▼
  aTile  bTile  biasTile        ← LEFT / RIGHT / BIAS 地址空间
    │       │       │
    │ TMATMUL (PIPE_M, Cube 引擎)
    ▼
  cTile                         ← ACC 地址空间
    │
    │ × iters (Split-K 累加)
    │
    │ TSTORE (PIPE_FIX)
    ▼
  GM（全局内存）
  C[M,N]
```

---

## Python API → PTO Op 映射

| Python 调用 | 生成的 PTO Op | .pto 中示例行 |
|---|---|---|
| `pto.MakeTensorViewOp()` | `pto.make_tensor_view` | 行 13–16 |
| `pto.AllocTileOp()` | `pto.alloc_tile` | 行 17–23 |
| `pto.PartitionViewOp()` | `pto.partition_view` | 行 26–28, 60 |
| `pto.TLoadOp()` | `pto.tload` | 行 29–32 |
| `pto.record_event()` | `pto.set_flag` | 行 35, 43, 55, 58 |
| `pto.wait_event()` | `pto.wait_flag` | 行 36, 44, 56, 59 |
| `pto.TMovOp()` | `pto.tmov` | 行 37–41 |
| `pto.TMatmulOp()` | `pto.tmatmul` | 行 50 |
| `pto.TMatmulBiasOp()` | `pto.tmatmul.bias` | 行 48 |
| `pto.TMatmulAccOp()` | `pto.tmatmul.acc` | 行 53 |
| `pto.TStoreOp()` | `pto.tstore` | 行 61 |

---

## 关键设计点

1. **Split-K 策略**：K=256 切成 8 次 × 32，L1 缓冲区只需 32×32 tile 而非 32×256，降低片上存储压力。

2. **三重管道同步**：每对相邻管道（MTE2↔MTE1↔M↔MTE2）间插入 set/wait_flag，确保数据就绪后才启动下一步，利用 Ascend AICORE 的硬件事件机制实现零开销同步。

3. **条件 bias**：`isBias` 布尔参数通过 `scf.if` 实现运行时决策，避免为无 bias 场景生成冗余的 TLOAD/TMOV。

4. **首次/后续区分**：第一次迭代清零累加器（`tmatmul`/`tmatmul.bias`），后续迭代累加（`tmatmul.acc`），保证 Split-K 的数学正确性。

5. **Tile 配置对齐硬件**：LEFT 用 `ColMajor`，RIGHT 用 `RowMajor+SLayout=ColMajor`，ACC 用 `fractal=1024`，这些都与 pto-isa C++ 库的 Tile 模板参数一一对应。
