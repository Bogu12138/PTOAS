# PTO IR 能力全景

本文档全面梳理 PTO IR 的能力边界：类型系统、地址空间、硬件管道、202 个操作、29 个枚举、2 个接口，以及关键设计模式。

---

## 概要统计

| 类别 | 数量 |
|---|---|
| **PTO Types** | 12 |
| **PTO Enums** | 29 |
| **PTO Attributes** | 31 |
| **PTO Interfaces** | 2 |
| **PTO Operations** | 202 |

定义位置：`include/PTO/IR/` 下的 `.td` 文件，由 TableGen 生成 `.h.inc` / `.cpp.inc`。

---

## 1. 类型系统

| 类型 | 助记符 | 参数 | 用途 |
|---|---|---|---|
| `PtrType` | `!pto.ptr<elem>` | 元素类型 | 类型化指针，指向 GM 数据 |
| `TensorViewType` | `!pto.tensor_view<d0xd1x...xelem>` | 形状 + 元素类型 | GM 数据的逻辑视图（shape + stride），不拥有数据 |
| `PartitionTensorViewType` | `!pto.partition_tensor_view<...>` | 形状 + 元素类型 | TensorView 的逻辑切片，对应一次 TLOAD/TSTORE 的范围 |
| `TileType` | `!pto.tile<...>` | 形状 + 元素类型 | 轻量 tile 描述符 |
| `TileBufType` | `!pto.tile_buf<...>` | 形状 + 元素类型 + 地址空间 + 有效形状 + 配置 | 物理 tile 缓冲区，编码了完整的硬件 tile 信息 |
| `EventIdArrayType` | `!pto.eventid_array<N>` | 大小 | 动态事件 ID 本地数组，用于自定义同步 |
| `PipeType` | `!pto.pipe` | 无 | TPUSH/TPOP 管道的不透明句柄 |
| `AsyncSessionType` | `!pto.async_session` | 无 | 异步 DMA 会话句柄 |
| `AsyncEventType` | `!pto.async_event` | 无 | 异步 DMA 事件句柄 |
| `HiF8Type` | `!pto.hif8` | 无 | A5 HiFloat8（8 位自定义浮点），实现 DataLayoutTypeInterface |
| `F4E1M2x2Type` | `!pto.f4E1M2x2` | 无 | 打包 FP4 对（E1M2 格式，2-per-byte） |
| `F4E2M1x2Type` | `!pto.f4E2M1x2` | 无 | 打包 FP4 对（E2M1 格式，2-per-byte） |

### TileBufType 详解

TileBufType 是 PTO IR 最重要的类型，编码了完整的硬件 tile 描述：

```
!pto.tile_buf<loc=left, dtype=f32, rows=32, cols=32,
               v_row=32, v_col=32, blayout=col_major,
               slayout=row_major, fractal=512, pad=0>
```

| 参数 | 含义 |
|---|---|
| `loc` (地址空间) | LEFT / RIGHT / ACC / MAT / VEC / BIAS / SCALING |
| `dtype` | 元素类型：f16 / f32 / bf16 / s8 / u8 / hif8 等 |
| `rows, cols` | tile 的逻辑形状 |
| `v_row, v_col` | 有效数据形状（用于 padding 场景） |
| `blayout` | 基础布局：RowMajor / ColMajor |
| `slayout` | 次级布局：NoneBox / RowMajor / ColMajor |
| `fractal` | 分形大小：512 (AB) / 1024 (C) |
| `pad` | 填充策略：Null / Zero / Max / Min |

TileBufConfigAttr 由 BLayout + SLayout + fractal + pad + compactMode 组合而成，与 pto-isa C++ 库的 Tile 模板参数一一对应。

---

## 2. 地址空间与存储层次

地址空间直接映射 Ascend AICORE 的物理存储层次：

```
┌─────────────────────────────────────────────┐
│  GM (Global Memory)                         │  ← 片外，存放输入/输出张量
├─────────────────────────────────────────────┤
│  MAT                                        │  ← L1 搬运缓冲区
├─────────────────────────────────────────────┤
│  LEFT     │  RIGHT     │  BIAS              │  ← L0A/L0B/Bias
├─────────────────────────────────────────────┤
│  ACC                                        │  ← L0C 累加缓冲区（Cube 输出）
├─────────────────────────────────────────────┤
│  VEC                                        │  ← UB 向量缓冲区
├─────────────────────────────────────────────┤
│  SCALING                                    │  ← 量化参数缓冲区
└─────────────────────────────────────────────┘
```

| 枚举值 | 名称 | 含义 | 典型用途 |
|---|---|---|---|
| 1 | `GM` | 全局内存（片外） | 输入/输出张量、权重 |
| 2 | `MAT` | L1 矩阵缓冲区 | TLOAD 的中间搬运区 |
| 3 | `LEFT` | L0A 左缓冲区 | Cube 引擎的 A 输入 |
| 4 | `RIGHT` | L0B 右缓冲区 | Cube 引擎的 B 输入 |
| 5 | `ACC` | L0C 累加缓冲区 | Cube 引擎的计算结果 |
| 6 | `VEC` | UB 向量缓冲区 | Vector 引擎的计算区 |
| 7 | `BIAS` | 偏置缓冲区 | 偏置数据 |
| 8 | `SCALING` | 缩放缓冲区 | 量化/反量化参数 |

---

## 3. 硬件管道

每个计算/搬运操作声明其在哪个硬件管道上执行（通过 `OpPipeInterface`）：

| 管道 | 硬件单元 | 典型操作 |
|---|---|---|
| `PIPE_S` | 标量单元 | tgetval, tsetval, tscatter (A3) |
| `PIPE_V` | 向量处理单元 | tadd, trelu, tcvt, tcmp, treduce... |
| `PIPE_M` | 矩阵（Cube）乘法单元 | tmatmul, tgemv |
| `PIPE_MTE1` | 内存搬运引擎 1 | tmov (MAT → L0), textract, tinsert |
| `PIPE_MTE2` | 内存搬运引擎 2 | tload (GM → 本地) |
| `PIPE_MTE3` | 内存搬运引擎 3 | tstore (非 ACC), mscatter |
| `PIPE_MTE4` | 内存搬运引擎 4 | 预留 |
| `PIPE_MTE5` | 内存搬运引擎 5 | 预留 |
| `PIPE_V2` | 向量单元 2 | 预留 |
| `PIPE_FIX` | 定点/转换单元 | tstore (ACC → GM), tmov (ACC → VEC), textract_fp |

管道声明驱动自动同步插入：`SyncInsertion` pass 根据相邻管道间插入 `set_flag` / `wait_flag`。

---

## 4. 操作分类总览

202 个操作按功能分为以下类别：

| 类别 | 数量 | 说明 |
|---|---|---|
| 指针/视图/内存管理 | 12 | 指针运算、视图创建、tile 分配与绑定 |
| DMA / 数据搬运 | 7 | TLOAD、TSTORE、TMOV、转置、预取 |
| 系统查询 | 4 | 核心号、核心数量 |
| 高层同步 | 3 | record_event、wait_event、barrier_sync |
| 低层同步 | 11 | set_flag、wait_flag、sync.set/wait、barrier、tsync、get_buf/rls_buf |
| Section | 2 | Cube/Vector 代码段宏守卫 |
| 前端管道通信 | 12 | Cube↔Vector 的 alloc/push/pop/free |
| 统一管道操作 | 13 | initialize_pipe、tpush、tpop、tfree、declare 等 |
| 集合通信 | 14 | 异步 DMA、同步读写、信号通知、广播/聚合/散播/规约 |
| 矩阵乘法族 | 6 | tmatmul + acc/bias/mx 变体 |
| 矩阵-向量乘法族 | 6 | tgemv + acc/bias/mx 变体 |
| 逐元素一元 | 13 | abs, exp, log, sqrt, relu, neg, cvt... |
| 逐元素二元 | 14 | add, sub, mul, div, max, min, and, or, xor, shl, shr, cmp... |
| 逐元素三元 | 2 | taddc, tsubc |
| 标量-Tile | 14 | adds, subs, muls, divs, maxs, mins, ands, ors, xors, shls, shrs, cmps, rems, fmods |
| 融合标量-Tile-Tile | 3 | taddsc, taxpy, tsubsc |
| 列方向广播/规约 | 14 | tcolexpand{,add,mul,div,sub,expdif,max,min}, tcol{max,argmax,min,argmin,sum,prod} |
| 行方向广播/规约 | 14 | trowexpand{,add,mul,div,sub,expdif,max,min}, trow{max,argmax,min,argmin,sum,prod} |
| Gather/Scatter/索引 | 5 | mgather, mscatter, tgather, tgatherb, tscatter |
| Tile 操作 | 10 | tconcat, tconcatidx, textract, tinsert, tfillpad... |
| 选择/排序/特殊 | 4 | tsel, tsels, tsort32, tmrgsort |
| Partial 操作 | 6 | tpart{add,max,min,argmax,argmin,mul} |
| 量化 | 2 | tquant, tdequant |
| 其他 Tile 操作 | 7 | tmov.fp, tstore_fp, tprelu, tci, ttri, tsetval, tgetval... |
| 调试/工具 | 4 | print, trap, tprint, tget_scale_addr |
| 事件 ID 数组 | 3 | declare_eventid_array, get, set |

---

## 5. 操作详解

### 5.1 指针 / 视图 / 内存管理

| Op | 说明 | 管道 |
|---|---|---|
| `addptr` | 指针偏移运算 `ptr + offset` | Pure |
| `make_tensor_view` | 将指针封装为 TensorView（shape + stride） | — |
| `partition_view` | 从 TensorView 切出逻辑切片 | — |
| `get_tensor_view_dim` | 查询 TensorView 维度大小 | Pure |
| `alloc_tile` | 分配 tile 缓冲区（可带可选地址） | — |
| `bind_tile` | 将配置绑定到 memref | Pure |
| `materialize_tile` | 从内存规划后的 memref 具象化 tile 句柄 | Pure |
| `subview` | 从父 tile 创建子视图 | Pure |
| `set_validshape` | 运行时更新 tile 的有效形状 | — |
| `bitcast` | SSA 类型重解释（别名原存储） | Pure |
| `pointer_cast` | 整数地址转 MemRef | Pure |
| `tassign` | 重绑定 tile 句柄到运行时地址 | Pure |

### 5.2 DMA / 数据搬运

| Op | 说明 | 管道 | 方向 |
|---|---|---|---|
| `tload` | 从 PartitionView 加载数据到 tile_buf | MTE2 | GM → 本地 |
| `tprefetch` | 预取全局数据到本地 tile 缓冲区 | MTE2 | GM → 本地 |
| `tstore` | 从 tile_buf 写回 PartitionView | FIX/MTE3 | 本地 → GM |
| `ttrans` | 矩阵转置 | V | — |
| `tmov` | 在地址空间间搬运数据 | 动态 | MAT↔L0, ACC↔VEC 等 |
| `tmov.fp` | 带缩放 tile 的搬运/转换 | MTE1 | — |
| `textract` / `textract_fp` | 从大 tile 中提取子窗口 | 动态/FIX | — |
| `tinsert` / `tinsert_fp` | 将子 tile 插入大 tile | 动态/FIX | — |
| `tfillpad` / `tfillpad_expand` / `tfillpad_inplace` | 带填充的复制 | V | — |

TLOAD 支持可选的 padding（`pad_mode`、`pad_value`、`left/right_padding_num`），以及 `init_out_buffer` 初始化选项。

TSTORE 支持多种模式：
- `st_phase`：存储阶段（Unspecified / Partial / Final）
- `atomic_type`：原子操作（None / Add）
- `relu_pre_mode`：ReLU 预处理（NoRelu / NormalRelu）
- `preQuantScalar`：预量化标量

### 5.3 同步机制

PTO IR 提供三层同步抽象：

**高层同步**（基于操作类型，由 `LoweringSyncToPipe` pass 降级为底层）：

| Op | 说明 |
|---|---|
| `record_event` | 记录同步事件（TLOAD→MTE2, TMATMUL→M 等） |
| `wait_event` | 等待同步事件 |
| `barrier_sync` | 高层屏障 |

**底层同步**（直接操作管道和事件 ID）：

| Op | 说明 |
|---|---|
| `set_flag` / `set_flag_dyn` | 在管道间设置同步标志（静态/动态事件 ID） |
| `wait_flag` / `wait_flag_dyn` | 等待管道间同步标志 |
| `sync.set` / `sync.wait` | Cube↔Vector 核间同步信号 |
| `barrier` | 管道内内存屏障 |
| `tsync` | 直接映射硬件 TSYNC 指令 |

**Buffer-ID 同步**（A5 专用）：

| Op | 说明 |
|---|---|
| `get_buf` | 获取 buffer-id 令牌用于操作排序 |
| `rls_buf` | 释放 buffer-id 令牌 |

典型同步流（以 Split-K MatMul 为例）：

```
TLOAD → set_flag[MTE2→MTE1] → wait_flag → TMOV → set_flag[MTE1→M] → wait_flag → TMATMUL → set_flag[M→MTE2] → ...
```

### 5.4 矩阵乘法族

所有矩阵乘法使用 DPS（目的传递风格）：

| Op | 操作数 | 说明 |
|---|---|---|
| `tmatmul` | A, B → C | 基础矩阵乘法（L0C 清零） |
| `tmatmul.acc` | C_in, A, B → C | 累加矩阵乘法（L0C 保留） |
| `tmatmul.bias` | A, B, bias → C | 带偏置矩阵乘法 |
| `tmatmul.mx` | A, a_scale, B, b_scale → C | MX 混合精度矩阵乘法 |
| `tmatmul.mx.acc` | C_in, A, a_scale, B, b_scale → C | MX 累加 |
| `tmatmul.mx.bias` | A, a_scale, B, b_scale, bias → C | MX 带偏置 |

### 5.5 矩阵-向量乘法族

与矩阵乘法结构相同，但针对矩阵-向量乘法优化：

| Op | 说明 |
|---|---|
| `tgemv` | 基础矩阵-向量乘法 |
| `tgemv.acc` | 累加 |
| `tgemv.bias` | 带偏置 |
| `tgemv.mx` | 混合精度 |
| `tgemv.mx.acc` | MX 累加 |
| `tgemv.mx.bias` | MX 带偏置 |

### 5.6 逐元素计算

#### 一元运算（13 个）

| Op | 说明 |
|---|---|
| `tabs` | 绝对值 |
| `texp` | 指数 |
| `tlog` | 自然对数 |
| `tsqrt` | 平方根 |
| `trsqrt` | 平方根倒数（可选 tmp） |
| `trecip` | 倒数 |
| `tneg` | 取反 |
| `tnot` | 按位取反 |
| `trelu` | ReLU |
| `tlrelu` | Leaky ReLU（标量斜率） |
| `tprelu` | PReLU（逐元素斜率 tile） |
| `tcvt` | 类型转换（可配置舍入/饱和模式） |
| `texpands` | 标量广播到 tile |

`tcvt` 支持丰富的转换模式：
- **舍入模式**：NONE, RINT, ROUND, FLOOR, CEIL, TRUNC, ODD, CAST_RINT
- **饱和模式**：ON, OFF

#### 二元运算（14 个）

算术：`tadd`, `tsub`, `tmul`, `tdiv`, `tfmod`, `trem`
比较：`tmax`, `tmin`, `tcmp`
位运算：`tand`, `tor`, `txor`, `tshl`, `tshr`

#### 三元运算（2 个）

- `taddc`：`dst = src0 + src1 + src2`
- `tsubc`：`dst = src0 - src1 + src2`

#### 标量-Tile 运算（14 个）

与二元运算一一对应，第二个操作数为标量：`tadds`, `tsubs`, `tmuls`, `tdivs`, `tfmods`, `trems`, `tmaxs`, `tmins`, `tands`, `tors`, `txors`, `tshls`, `tshrs`, `tcmps`

#### 融合运算（3 个）

- `taddsc`：`dst = src0 + scalar + src1`
- `tsubsc`：`dst = src0 - scalar + src1`
- `taxpy`：`dst += src * scalar`（AXPY）

### 5.7 规约与广播

#### 列方向（14 个）

广播（将每列第一个元素扩展到整列）：
`tcolexpand`, `tcolexpandadd`, `tcolexpandmul`, `tcolexpanddiv`, `tcolexpandsub`, `tcolexpandexpdif`, `tcolexpandmax`, `tcolexpandmin`

规约（沿行方向归约，结果为每列一个值）：
`tcolmax`, `tcolargmax`, `tcolmin`, `tcolargmin`, `tcolsum`, `tcolprod`

#### 行方向（14 个）

广播（将每行第一个元素扩展到整行）：
`trowexpand`, `trowexpandadd`, `trowexpandmul`, `trowexpanddiv`, `trowexpandsub`, `trowexpandexpdif`, `trowexpandmax`, `trowexpandmin`

规约（沿列方向归约，结果为每行一个值）：
`trowmax`, `trowargmax`, `trowmin`, `trowargmin`, `trowsum`, `trowprod`

### 5.8 Gather / Scatter / 选择

| Op | 说明 | 管道 |
|---|---|---|
| `mgather` | 按索引从内存 gather-load 到 tile | MTE2 |
| `mscatter` | 按索引从 tile scatter-store 到内存 | MTE3 |
| `tgather` | 多形式 gather（索引/掩码/标量比较） | V |
| `tgatherb` | 按字节偏移 gather | V |
| `tscatter` | 按索引 scatter | S (A3) / V (A5) |
| `tsel` | 按掩码在两个 tile 间逐元素选择 | V |
| `tsels` | 按标量选择模式全局选择 | V |

### 5.9 Tile 操作

| Op | 说明 |
|---|---|
| `tconcat` | 沿列维度拼接两个 tile |
| `tconcatidx` | 带逐行索引控制的拼接 |
| `textract` / `textract_fp` | 从大 tile 提取子窗口 |
| `tinsert` / `tinsert_fp` | 将子 tile 插入大 tile |
| `tfillpad` / `tfillpad_expand` / `tfillpad_inplace` | 带填充的复制 |
| `treshape` | 重解释 tile 视图（别名存储） |

### 5.10 排序 / 特殊计算

| Op | 说明 |
|---|---|
| `tsort32` | 固定 32 元素块排序 |
| `tmrgsort` | 归并排序（1-4 个源 tile） |
| `thistogram` | 逐行 256-bin 直方图累加 |
| `tci` | 生成连续整数序列（可逆序） |
| `ttri` | 生成三角掩码 tile |
| `trandom` | 随机数生成（key/counter 输入） |

### 5.11 Partial 操作（有效区域感知）

用于不同 tile 有效区域不匹配的场景：

`tpartadd`, `tpartmax`, `tpartmin`, `tpartargmax`, `tpartargmin`, `tpartmul`

### 5.12 量化

| Op | 说明 |
|---|---|
| `tquant` | 量化 tile（INT8_SYM / INT8_ASYM），使用缩放 tile |
| `tdequant` | 反量化整数 tile，使用逐行缩放和偏移 tile |

### 5.13 核间通信 / 管道

**前端管道操作**（Cube↔Vector）：

| Op | 说明 |
|---|---|
| `aic_initialize_pipe` / `aiv_initialize_pipe` | Cube/Vector 核管道初始化 |
| `talloc_to_aiv` / `talloc_to_aic` | 生产端分配 FIFO 条目 |
| `tpush_to_aiv` / `tpush_to_aic` | 推送数据到对端核 |
| `tpop_from_aic` / `tpop_from_aiv` | 从对端核弹出数据 |
| `tfree_from_aic` / `tfree_from_aiv` | 释放消费者槽位 |
| `reserve_buffer` / `import_reserved_buffer` | 预留/导入缓冲区 |

**统一管道操作**：

| Op | 说明 |
|---|---|
| `initialize_l2g2l_pipe` / `initialize_l2l_pipe` | 初始化管道句柄 |
| `tpush` / `tpop` / `tfree` / `talloc` | 标准管道操作 |

### 5.14 集合通信

| Op | 说明 | 模式 |
|---|---|---|
| `comm.build_async_session` | 构建异步 DMA 会话 | — |
| `comm.tput_async` / `comm.tget_async` | 异步远程读/写 | 异步 |
| `comm.wait_async_event` / `comm.test_async_event` | 等待/测试异步完成 | 异步 |
| `comm.tput` / `comm.tget` | 同步远程读/写 | 同步 |
| `comm.tnotify` | 发送信号通知 | 同步 |
| `comm.twait` / `comm.ttest` | 等待/测试信号 | 同步/异步 |
| `comm.tbroadcast` | 广播到所有组成员 | 集合 |
| `comm.tgather` / `comm.tscatter` | 聚合/散播 | 集合 |
| `comm.treduce` | 跨组规约（Sum/Max/Min） | 集合 |

### 5.15 系统查询

| Op | 说明 | 返回 |
|---|---|---|
| `get_block_idx` | 当前核 ID | I64 |
| `get_subblock_idx` | 当前 Vector 核 ID (0 或 1) | I64 |
| `get_block_num` | 总核数 | I64 |
| `get_subblock_num` | Vector 核数量 | I64 |

### 5.16 Section 操作

| Op | 说明 |
|---|---|
| `section.cube` | Cube 核代码段，受 `#if defined(...)` 宏守卫 |
| `section.vector` | Vector 核代码段 |

### 5.17 调试 / 工具

| Op | 说明 |
|---|---|
| `print` | 调试打印（格式字符串 + 标量） |
| `tprint` | 从设备打印 tile/GlobalTensor 内容 |
| `trap` | 终止执行 |
| `tget_scale_addr` | 绑定缩放 tile 到源 tile 的缩放地址 |

---

## 6. 关键枚举

### 舍入模式（RoundMode）

| 值 | 含义 |
|---|---|
| NONE | 无舍入 |
| RINT | 就近舍入 |
| ROUND | 四舍五入 |
| FLOOR | 向下取整 |
| CEIL | 向上取整 |
| TRUNC | 截断 |
| ODD | 向奇数舍入 |
| CAST_RINT | 类型转换 + 就近舍入 |

### 比较模式（CmpMode）

EQ, NE, LT, LE, GT, GE

### TSTORE 存储阶段（STPhase）

| 值 | 含义 |
|---|---|
| Unspecified | 默认（单次存储） |
| Partial | 部分存储 |
| Final | 最终存储 |

### Gather/Scatter 越界模式

| OOB 模式 | 含义 |
|---|---|
| Undefined | 未定义行为 |
| Clamp | 钳制到合法范围 |
| Wrap | 环绕 |
| Zero | 返回零 / Skip |

---

## 7. 关键设计模式

### 7.1 DPS（目的传递风格）

绝大多数 tile 计算操作使用 `ins(...) outs(...)` 格式：

```
%result = pto.tmatmul ins(%lhs, %rhs: ...) outs(%dst: ...)
```

输出缓冲区作为操作数传入，而非作为结果产出。这与 MLIR 的 Linalg 方言类似，好处是：
- 支持原地更新（累加操作复用 ACC 缓冲区）
- 内存规划可以在编译期预分配所有缓冲区
- 与 `PTO_DpsInitOpInterface` 配合，支持内存复用优化

### 7.2 OpPipeInterface

几乎每个 tile 操作都实现了 `OpPipeInterface`，声明其执行的硬件管道。这驱动了 `SyncInsertion` pass 自动在不同硬件单元间插入同步屏障。

### 7.3 多架构支持

IR 同时支持 A3 (`Ascend910B1`) 和 A5 (`Ascend950`)。部分操作的管道映射因架构而异：

- `tscatter`：A3 上走 PIPE_S，A5 上走 PIPE_V
- `tpop`：A5 上走 PIPE_S，A3 上走 PIPE_MTE2
- Buffer-ID 同步（`get_buf`/`rls_buf`）为 A5 专有

架构通过 Module 级属性 `pto.target_arch` 标识。

### 7.4 核类型系统

| 核类型 | 含义 |
|---|---|
| `AIC` | Cube 核（矩阵乘法） |
| `AIV` | Vector 核（向量计算） |
| `MIX` | 混合模式 |
| `AIC_OR_AIV` | 自动选择 |

函数通过 `TFuncCoreTypeAttr` 标记核类型，Module 通过 `TModuleCoreTypeAttr` 推断整体核类型。`section.cube` / `section.vector` 操作用于在混合核中区分代码段。

### 7.5 布局系统

Tile 缓冲区的布局由两级控制：

- **BLayout**（基础布局）：RowMajor / ColMajor — 决定数据在 tile 中的排列方式
- **SLayout**（次级布局）：NoneBox / RowMajor / ColMajor — 控制分形内的子排列
- **CompactMode**：Null / Normal / RowPlusOne — 紧凑存储模式
- **PadValue**：Null / Zero / Max / Min — 填充策略

不同地址空间有典型的布局约定：
- LEFT（A 输入）：ColMajor + RowMajor
- RIGHT（B 输入）：RowMajor + ColMajor
- ACC（C 输出）：ColMajor + RowMajor + fractal=1024
- BIAS：RowMajor + NoneBox

---

## 8. 与主流 MLIR 方言对比

| 特性 | PTO Dialect | Linalg | TOSA | Affine |
|---|---|---|---|---|
| **抽象层次** | Tile 级硬件操作 | 张量/缓冲区级 | 算子级 | 循环级 |
| **目标硬件** | Ascend NPU (专用) | 通用 | 通用 | 通用 |
| **内存模型** | 显式地址空间 (GM→MAT→L0→ACC) | 隐式 | 隐式 | 显式 |
| **同步** | 硬件管道事件 (set_flag/wait_flag) | 无 | 无 | 无 |
| **计算单元** | Cube/Vector/MTE 管道 | 通用 | 通用 | 通用 |
| **数据类型** | 标准 + HiF8/F4 等自定义类型 | 标准 | 标准 | 标准 |
| **通信** | 核间管道 + 集合通信 | 无 | 无 | 无 |

PTO IR 的独特之处在于它是**硬件直射型 IR**——每个操作都直接映射到 Ascend AICORE 的特定硬件单元和管道，而非通过多层抽象逐步降低。
