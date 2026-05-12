# PTOAS Pass 全景

本文档介绍 ptoas 编译管线中的全部 Pass，包括每个 Pass 的功能、输入/输出、在管线中的位置，以及它们之间的依赖关系。

---

## Pass 总表

### 默认管线 Pass

| # | Pass 名称 | 粒度 | 阶段 | 功能 | CLI 控制 | 实现文件 |
|---|---|---|---|---|---|---|
| 1 | **PTOAssignDefaultFrontendPipeId** | FuncOp | 前端规范化 | 为省略 `id` 的前端管道操作补上默认 `id=0` | 始终执行 | `PTOAssignDefaultFrontendPipeIdPass.cpp` |
| 2 | **PTOLowerFrontendPipeOps** | FuncOp | 前端规范化 | 将前端管道 op（`aic_initialize_pipe`、`tpush_to_aiv` 等）降级为内部统一管道 IR（`initialize_l2l_pipe`、`tpush`、`tpop` 等） | 始终执行 | `PTOLowerFrontendPipeOpsPass.cpp` |
| 3 | **PTOInferValidatePipeInit** | ModuleOp | 前端规范化 | 推断和校验内部管道 `nosplit` 配置，传播到管道对端 | 始终执行 | `PTOInferValidatePipeInitPass.cpp` |
| 4 | **LoweringSyncToPipe** | FuncOp | 同步/布局 | 将 `record_event` / `wait_event` 降级为 `set_flag` / `wait_flag`，SyncOpType → PIPE 枚举映射 | 始终执行 | `LoweringSyncToPipe.cpp` |
| 5 | **InferPTOLayout** | FuncOp | 同步/布局 | 为 `make_tensor_view` 推断全局张量布局（ND/DN/NZ） | `--disable-infer-layout` 关闭 | `InferPTOLayout.cpp` |
| 6 | **PTOA5NormalizeTMov** | FuncOp | 同步/布局 | 规范化 A5 上不安全的 vec→vec col_major TMOV 为 row_major 路径 | 始终执行 | `PTOA5NormalizeTMovPass.cpp` |
| 7 | **PTOViewToMemref** | ModuleOp | 内存规划 | 将 TileBufType / TensorView 转为 memref，通过 `bind_tile` 保留 tile 元数据 | 始终执行 | `PTOViewToMemref.cpp` |
| 8 | **PlanMemory** | ModuleOp | 内存规划 | 为所有 tile 缓冲区规划物理内存地址，生命周期不重叠的 tile 复用内存 | `--pto-level=level3` 跳过 | `PTOPlanMemory.cpp` |
| 9 | **PTOResolveReservedBuffers** | ModuleOp | 内存规划 | 解析 `reserve_buffer` / `import_reserved_buffer` 为常量地址，对齐管道 `flag_base` | 始终执行 | `PTOResolveReservedBuffersPass.cpp` |
| 10a | **InsertSync** | FuncOp | 同步插入 | 分析内存依赖，在管道间插入最小 `set_flag` / `wait_flag` | `--enable-insert-sync` | `InsertSync/`（10 文件） |
| 10b | **InjectBarrierAllSync** | FuncOp | 同步插入 | 在每个有内存副作用的管道操作前插入 `barrier <PIPE_ALL>` | `--enable-inject-barrier-all-sync` | `PTOInjectBarrierAllSync.cpp` |
| 10c | **GraphSyncSolver** | FuncOp | 同步插入 | 构建冲突图，Dijkstra 求解 + 图着色分配事件 ID，生成优化同步方案 | `--enable-graph-sync-solver` | `GraphSyncSolver/`（8 文件） |
| 11 | **PTOMaterializeTileHandles** | ModuleOp | 代码生成 | 将内存规划后的 memref 重新包装为 `materialize_tile`，恢复 tile 句柄供 EmitC 使用 | 始终执行 | `PTOMaterializeTileHandles.cpp` |
| 12 | **CSE** | — | 代码生成 | 公共子表达式消除，清理 MaterializeTileHandles 的冗余 | 始终执行 | MLIR 内置 |
| 13 | **PTOToEmitC** | ModuleOp | 代码生成 | 202 个 PTO op → `emitc::CallOpaqueOp`，TileBufType → C++ Tile 模板 | `--pto-arch=a3/a5` | `PTOToEmitC.cpp`（11890 行） |
| 14 | **FormExpressions + CSE** | — | 代码生成 | 将分散的 EmitC 操作合并为 C++ 表达式树，消除冗余 | 始终执行 | MLIR 内置 |

### 非管线 Pass

| Pass 名称 | 粒度 | 功能 | 状态 |
|---|---|---|---|
| **PTOVerifyTFree** | FuncOp | 校验 `tpop` / `tfree` 配对正确性 | 默认管线中注释掉 |
| **PTOWrapFunctionsInSections** | FuncOp | 将 `kernel_kind=cube/vector` 的函数体包裹在 `section.cube/vector` 中 | 按需使用 |
| **ConvertToPTOOp** | — | 将其他方言操作转换为 PTO 操作 | 按需使用 |
| **InferPTOMemScope** | — | 推断和传播 PTO 操作的内存作用域信息 | 按需使用 |
| **PTORemoveRedundantBarrier** | — | 移除冗余 barrier 操作 | 按需使用 |
| **AllocToPointerCast** | — | 将 alloc 转换为 pointer_cast | 内部使用 |
| **OptMemPlanForPipeline** | — | 针对流水线场景的内存规划优化 | 内部使用 |

### 后处理（非 Pass）

| 步骤 | 功能 |
|---|---|
| `dropEmptyEmitCExpressions` | 清除 FormExpressions 产生的空节点 |
| `materializeControlFlowOperands` | 确保 scf 控制流操作数正确嵌入 EmitC |
| `reorderEmitCFunctions` | 按 C++ 编译依赖重排输出函数（被调用者在前） |

---

## 管线总览

```
输入: .pto (MLIR 文本) / .ptobc (二进制)
  │
  ▼ Parse → ModuleOp
  │
  ├─ 1.  PTOAssignDefaultFrontendPipeId    ← 前端管道默认值
  ├─ 2.  PTOLowerFrontendPipeOps           ← 前端管道降级
  ├─ 3.  PTOInferValidatePipeInit          ← 管道初始化推断与校验
  ├─ 4.  LoweringSyncToPipe                ← 高层同步 → 底层同步
  ├─ 5.  InferPTOLayout                    ← 布局推断（可选）
  ├─ 6.  PTOA5NormalizeTMov                ← A5 TMOV 规范化
  ├─ 7.  PTOViewToMemref                   ← 视图类型 → memref
  ├─ 8.  PlanMemory                        ← 内存规划（Level3 跳过）
  ├─ 9.  PTOResolveReservedBuffers         ← 预留缓冲区解析
  ├─10.  SyncInsertion (三选一)             ← 同步插入
  │      ├─ InsertSync
  │      ├─ InjectBarrierAllSync
  │      └─ GraphSyncSolver
  ├─11.  PTOMaterializeTileHandles          ← tile 句柄具象化
  ├─12.  CSE                                ← 公共子表达式消除
  ├─13.  PTOToEmitC                         ← PTO → EmitC 降级
  ├─14.  FormExpressions + CSE              ← 表达式折叠
  │
  ▼ 后处理 → C++ 输出
```

---

## Phase 1：前端规范化（Pass 1–3）

### 1. PTOAssignDefaultFrontendPipeId

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:205` |
| **实现** | `lib/PTO/Transforms/PTOAssignDefaultFrontendPipeIdPass.cpp` |
| **粒度** | FuncOp |
| **功能** | 为前端管道操作补上默认 `id = 0`。旧版前端管道语法（`aic_initialize_pipe`、`tpush_to_aiv` 等）可能省略 `id` 属性，此 Pass 将缺失的 `id` 重写为显式的 `id = 0` |
| **目的** | 向后兼容，确保后续 Pass 不需要处理缺失属性 |

### 2. PTOLowerFrontendPipeOps

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:188` |
| **实现** | `lib/PTO/Transforms/PTOLowerFrontendPipeOpsPass.cpp` |
| **粒度** | FuncOp |
| **功能** | 将前端管道操作降级为内部统一管道 IR |

转换映射：

| 前端 Op | → 内部 Op |
|---|---|
| `aic_initialize_pipe` / `aiv_initialize_pipe` | `initialize_l2l_pipe` / `initialize_l2g2l_pipe` |
| `talloc_to_aiv` / `talloc_to_aic` | `declare_tile` + `talloc` |
| `tpush_to_aiv` / `tpush_to_aic` | `tpush` |
| `tpop_from_aic` / `tpop_from_aiv` | `tpop` |
| `tfree_from_aic` / `tfree_from_aiv` | `tfree` |

**意义**：前端管道 API 面向 Cube/Vector 核对（C2V/V2C），语义直观；内部统一管道 IR 更通用，支持本地到本地、本地到全局等多种模式，便于后续内存规划和同步分析。

### 3. PTOInferValidatePipeInit

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:222` |
| **实现** | `lib/PTO/Transforms/PTOInferValidatePipeInitPass.cpp` |
| **粒度** | ModuleOp |
| **功能** | 推断和校验内部管道初始化的 `nosplit` 配置 |

具体检查：
- 校验同一逻辑管道的下游操作不会混用 `split=0` 和 `split=1/2`
- 保留显式 `nosplit` 属性，拒绝冲突配置
- 为缺失 `nosplit` 的旧 IR 推断默认值
- 将解析结果传播到管道对端（生产端和消费端一致）

---

## Phase 2：同步与布局规范化（Pass 4–6）

### 4. LoweringSyncToPipe

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:156` |
| **实现** | `lib/PTO/Transforms/LoweringSyncToPipe.cpp` |
| **粒度** | FuncOp |
| **功能** | 将高层同步操作降级为底层管道同步操作 |

| 高层 Op | → 底层 Op |
|---|---|
| `record_event(src_op, dst_op, event_id)` | `set_flag(src_pipe, dst_pipe, event_id)` |
| `wait_event(src_op, dst_op, event_id)` | `wait_flag(src_pipe, dst_pipe, event_id)` |
| `barrier_sync(op_type)` | `barrier(pipe)` |

`src_op` / `dst_op`（SyncOpType 枚举）通过 `mapSyncOpTypeToPipe()` 映射为 PIPE 枚举。映射定义在 `lib/PTO/IR/PTOSyncUtils.cpp:25`：

| SyncOpType | → PIPE |
|---|---|
| TLOAD | PIPE_MTE2 |
| TSTORE_ACC | PIPE_FIX |
| TSTORE_VEC | PIPE_MTE3 |
| TMOV_M2L | PIPE_MTE1 |
| TMATMUL | PIPE_M |
| TVEC | PIPE_V |
| ... | ... |

### 5. InferPTOLayout

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:99` |
| **实现** | `lib/PTO/Transforms/InferPTOLayout.cpp` |
| **粒度** | FuncOp |
| **控制** | 默认启用，`--disable-infer-layout` 关闭 |
| **功能** | 为 `make_tensor_view` 推断全局张量布局（ND/DN/NZ） |

根据静态 shape 和 stride，按设计规则计算布局属性并附加到 `make_tensor_view` 的 `layout` 属性上。动态 stride/shape 不处理。

### 6. PTOA5NormalizeTMov

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:110` |
| **实现** | `lib/PTO/Transforms/PTOA5NormalizeTMovPass.cpp` |
| **粒度** | FuncOp |
| **功能** | 规范化 A5 上不安全的 vec→vec col_major TMOV |

重写模式：
```
src(col_major) → treshape(row_major) → tmov(row→row) → treshape(col_major) ← dst
```

通过 SSA `treshape`（纯视图重解释，无实际数据搬运）将不支持的 col_major TMOV 路径转为安全的 row_major 路径。

---

## Phase 3：视图降级与内存规划（Pass 7–9）

### 7. PTOViewToMemref

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:284` |
| **实现** | `lib/PTO/Transforms/PTOViewToMemref.cpp` |
| **粒度** | ModuleOp |
| **功能** | 将 PTO 视图/Tile 操作降级为 memref IR，同时通过绑定操作保留 tile 元数据 |

将 `TileBufType` 转换为 `MemRefType`，通过 `bind_tile` 将 Tile 配置信息绑定到 memref 上，使得后续的内存规划和同步分析可以在标准 memref IR 上进行。

**意义**：memref 是 MLIR 的标准内存抽象，有丰富的分析和变换基础设施。将 tile 视图转为 memref 使得 PlanMemory 和 SyncInsertion 可以利用 MLIR 的 `MemoryEffects` 接口和别名分析。

### 8. PlanMemory

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:130` |
| **实现** | `lib/PTO/Transforms/PTOPlanMemory.cpp` |
| **粒度** | ModuleOp |
| **控制** | `--pto-level=level3` 跳过此 Pass |
| **功能** | 为所有 tile 缓冲区规划物理内存地址 |

选项：
- `mem-plan-mode`：`local-mem-plan`（默认，memref.alloc）或 `global-workspace-plan`（全局工作区分配）
- `enable-global-workspace-reuse`：启用全局工作区复用
- `enable-print-memory-allocated-size`：打印分配大小
- `restrict-inplace-as-isa`：限制原地操作符合 ISA 约束

内存规划核心逻辑：
1. 分析 tile 的地址空间（LEFT/RIGHT/ACC/VEC 等）和生命周期
2. 对同一地址空间内生命周期不重叠的 tile 进行**内存复用**
3. 为 ping-pong 缓冲区排列连续地址（`ReorderContinuousPingPongEntry`）
4. 生成 `memref.alloc` 或全局工作区分配

### 9. PTOResolveReservedBuffers

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:245` |
| **实现** | `lib/PTO/Transforms/PTOResolveReservedBuffersPass.cpp` |
| **粒度** | ModuleOp |
| **前置** | PlanMemory |
| **功能** | 解析预留缓冲区地址和管道标志基址 |

具体工作：
- 对齐内部管道初始化的 `flag_base` 属性
- 校验无法追溯到 `reserve_buffer` / `import_reserved_buffer` 的管道
- 将 `reserve_buffer` / `import_reserved_buffer` 的结果替换为解析后的常量地址
- 擦除前端专用的 reserve/import 操作

---

## Phase 4：同步插入（Pass 10，三选一）

三者互斥，通过 CLI 标志选择。

### 10a. InsertSync

| 项目 | 内容 |
|---|---|
| **标志** | `--enable-insert-sync` |
| **定义** | `include/PTO/Transforms/Passes.td:24` |
| **实现** | `lib/PTO/Transforms/InsertSync/`（10 个文件） |
| **粒度** | FuncOp |
| **功能** | 分析 Cube/Vector/MTE 管道间的数据依赖，插入显式同步指令 |

内部流程：
1. **PTOIRTranslator**：将 PTO IR 翻译为内部同步分析 IR
2. **MemoryDependentAnalyzer**：分析内存依赖关系
3. **InsertSyncAnalysis**：基于内存冲突分析，确定需要插入同步的位置
4. **MoveSyncState**：处理循环携带的同步状态
5. **SyncEventIdAllocation**：分配事件 ID
6. **RemoveRedundantSync**：移除冗余同步
7. **SyncCodegen**：生成最终的 `set_flag` / `wait_flag` 操作

特点：保守但正确，基于内存冲突分析决定同步位置。

### 10b. InjectBarrierAllSync

| 项目 | 内容 |
|---|---|
| **标志** | `--enable-inject-barrier-all-sync` |
| **定义** | `include/PTO/Transforms/Passes.td:41` |
| **实现** | `lib/PTO/Transforms/PTOInjectBarrierAllSync.cpp` |
| **粒度** | FuncOp |
| **功能** | 在每个有内存副作用的管道操作前插入 `barrier <PIPE_ALL>` |

最保守策略：不分析依赖，直接在所有管道操作边界插入全屏障。用于调试或确保正确性的兜底方案。

### 10c. GraphSyncSolver

| 项目 | 内容 |
|---|---|
| **标志** | `--enable-graph-sync-solver` |
| **定义** | `include/PTO/Transforms/Passes.td:65` |
| **实现** | `lib/PTO/Transforms/GraphSyncSolver/`（8 个文件） |
| **粒度** | FuncOp |
| **功能** | 基于图的核内同步求解器（从 bishengir 移植） |

内部流程：
1. **SyncSolverIRTranslator**：将 PTO IR 翻译为层次化 SyncSolver IR（Function → Scope → Op）
2. **GraphSolver**：构建操作冲突图，用 Dijkstra 算法求解最短路径同步方案
3. **EventIdSolver**：通过图着色算法分配有限的事件 ID 槽位（最多 8 个）
4. **SyncSolverCodeGen**：生成最终的 `set_flag` / `wait_flag` / `barrier` 操作

选项：`--graph-sync-solver-event-id-max=N`（默认 8，限制可用事件 ID 数量）

特点：最智能的策略，通过可达性分析跳过被传递覆盖的同步候选，用图着色优化事件 ID 分配。

---

## Phase 5：代码生成（Pass 11–14）

### 11. PTOMaterializeTileHandles

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:301` |
| **实现** | `lib/PTO/Transforms/PTOMaterializeTileHandles.cpp` |
| **粒度** | ModuleOp |
| **功能** | 将内存规划后的 memref 重新包装为 `materialize_tile` 操作 |

**意义**：PlanMemory 和 SyncInsertion 在 memref IR 上工作，但 PTOToEmitC 需要 tile 句柄来生成 C++ Tile 模板代码。此 Pass 在两者之间搭建桥梁——将 memref 转回 tile 抽象，同时保留内存规划的结果。

### 12. CSE（第一轮）

标准 MLIR 公共子表达式消除 Pass，清理 MaterializeTileHandles 可能产生的冗余操作。

### 13. PTOToEmitC

| 项目 | 内容 |
|---|---|
| **实现** | `lib/PTO/Transforms/PTOToEmitC.cpp`（~11890 行） |
| **粒度** | ModuleOp |
| **控制** | `--pto-arch=a3` 或 `--pto-arch=a5` |
| **功能** | 将 PTO IR 的每个操作逐个转换为 EmitC 方言的 C++ 函数调用 |

核心转换：
- **类型**：`TileBufType` → C++ `Tile<TileType, dtype, rows, cols, BLayout, ...>` 模板字符串
- **操作**：202 个 PTO op 各有独立的 `RewritePattern`，转换为 `emitc::CallOpaqueOp`
- **属性**：PTO 枚举 → C++ 枚举字面量

### 14. FormExpressions + CSE（第二轮）

`emitc::createFormExpressionsPass()` 将分散的 EmitC 操作合并为 C++ 表达式树，减少临时变量。例如：

```
// 合并前
auto tmp0 = add(a, b);
auto tmp1 = mul(tmp0, c);
// 合并后
auto tmp1 = mul(add(a, b), c);
```

第二轮 CSE 清理表达式折叠后的冗余。

---

## 后处理（非 Pass）

在 Pass 管线之后，`ptoas.cpp` 还执行三个后处理步骤：

| 步骤 | 函数 | 功能 |
|---|---|---|
| `dropEmptyEmitCExpressions` | 清除空表达式 | 移除 FormExpressions 产生的空节点 |
| `materializeControlFlowOperands` | 物化控制流操作数 | 确保 scf 控制流的操作数正确嵌入 EmitC |
| `reorderEmitCFunctions` | 重排函数顺序 | 按 C++ 编译依赖重排输出函数（被调用者在前） |

---

## 其他 Pass（不在默认管线中）

### PTOVerifyTFree

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:268` |
| **实现** | `lib/PTO/Transforms/PTOVerifyTFreePass.cpp` |
| **状态** | 默认管线中注释掉（`//pm.addNestedPass<...>(createPTOVerifyTFreePass())`） |
| **功能** | 校验 `tpop` / `tfree` 的配对正确性 |

检查：
- 每个 `tpop` 在同一 block 中有匹配的 `tfree`
- 借用的 tile 在 `tfree` 之后不再被使用
- 同一管道不会在 `tfree` 之前累积多个未释放的 `tpop`

### PTOWrapFunctionsInSections

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:170` |
| **实现** | `lib/PTO/Transforms/PTOWrapFunctionsInSectionsPass.cpp` |
| **功能** | 将带 `kernel_kind` 属性的函数体包裹在 `section.cube` / `section.vector` 中 |

- `kernel_kind = cube` → 函数体包裹在 `pto.section.cube`
- `kernel_kind = vector` → 函数体包裹在 `pto.section.vector`

### ConvertToPTOOp

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:91` |
| **实现** | `lib/PTO/Transforms/ConvertToPTOOp.cpp` |
| **功能** | 将其他方言的操作转换为 PTO 操作 |

### InferPTOMemScope

| 项目 | 内容 |
|---|---|
| **定义** | `include/PTO/Transforms/Passes.td:124` |
| **实现** | `lib/PTO/Transforms/InferPTOMemScope.cpp` |
| **功能** | 推断和传播 PTO 操作的内存作用域信息 |

### PTORemoveRedundantBarrier

| 项目 | 内容 |
|---|---|
| **实现** | `lib/PTO/Transforms/PTORemoveRedundantBarrier.cpp` |
| **功能** | 移除冗余的 barrier 操作 |

---

## Pass 依赖关系图

```
PTOAssignDefaultFrontendPipeId ──→ PTOLowerFrontendPipeOps ──→ PTOInferValidatePipeInit
                                                                        │
                                                                        ▼
                                                              LoweringSyncToPipe
                                                                        │
                                                              InferPTOLayout (可选)
                                                                        │
                                                              PTOA5NormalizeTMov
                                                                        │
                                                              PTOViewToMemref
                                                                        │
                                                              PlanMemory ──→ PTOResolveReservedBuffers
                                                                        │
                                                              SyncInsertion (三选一)
                                                                        │
                                                              PTOMaterializeTileHandles
                                                                        │
                                                              CSE ──→ PTOToEmitC ──→ FormExpressions + CSE
```

关键依赖：
- **PlanMemory** 依赖 **PTOViewToMemref**（需要在 memref 上操作）
- **SyncInsertion** 依赖 **PlanMemory**（需要知道内存地址来判断冲突）
- **PTOToEmitC** 依赖 **PTOMaterializeTileHandles**（需要 tile 句柄而非 memref）
- **PTOResolveReservedBuffers** 依赖 **PlanMemory**（需要已规划的地址）

---

## 总结

PTOAS 的 Pass 管线可以按职责分为 5 个阶段：

| 阶段 | Pass | 职责 |
|---|---|---|
| **前端规范化** | 1–3 | 将框架特定的管道 API 转为通用内部 IR |
| **同步与布局** | 4–6 | 同步降级、布局推断、TMOV 规范化 |
| **内存规划** | 7–9 | 视图转 memref、地址分配、缓冲区解析 |
| **同步插入** | 10 | 根据依赖分析插入硬件同步指令 |
| **代码生成** | 11–14 | tile 具象化、PTO→EmitC 降级、表达式折叠 |

整个管线的核心思想是：**先将高层抽象逐步降级到接近硬件的 IR，再在低层 IR 上做精确的内存规划和同步分析，最后生成 C++ 代码**。
