# PTO IR 的两种格式：.pto 与 .ptobc

## 概要

`.pto` 和 `.ptobc` 是同一个 PTO IR 的两种序列化格式，完全等价。`ptoas` 都能直接消费。

| | `.pto` | `.ptobc` |
|---|---|---|
| **格式** | MLIR 文本（人类可读） | 自定义二进制（紧凑编码） |
| **内容** | 完整的 PTO ModuleOp | 完整的 PTO ModuleOp（编码后） |
| **大小** | 较大（例如 ~2KB） | 较小（~200-500B，约为文本的 1/5） |
| **可读性** | 可直接阅读和编辑 | 不可读 |
| **生产者** | Python `print(module)` / `ptoas --emit-pto-ir` | `ptobc encode` |
| **消费者** | `ptoas`（MLIR 文本解析器） | `ptoas`（自动检测魔术字节 `"PTOBC\0"` 解码） |

---

## 转换关系

```
.pto (文本)
  │
  ├──→ ptobc encode ──→ .ptobc (二进制)
  │
  └──→ ptoas ──→ .cpp

.ptobc (二进制)
  │
  ├──→ ptobc decode ──→ .pto (文本，roundtrip)
  │
  └──→ ptoas ──→ .cpp (ptoas 自动检测输入格式)
```

两者都能直接输入 `ptoas` 编译为 C++。`ptoas` 通过魔术字节 `"PTOBC\0"` 自动判断输入格式（`tools/ptoas/ptoas.cpp:981`）：

```cpp
const bool isPTOBC = (buf.size() >= 6 && std::memcmp(buf.data(), "PTOBC\0", 6) == 0);
```

---

## .ptobc 二进制格式结构

```
┌────────────────────────────────┐
│ Magic: "PTOBC\0" (6 bytes)    │  ← 标识二进制格式
│ Version: uint16 LE             │  ← 当前 v0 (0x0000)
│ Flags: uint16 LE               │  ← 当前 0x0000
│ Payload size: uint32 LE        │
│ Payload bytes:                 │
│   Section 0x01: Strings table  │  ← 去重字符串池
│   Section 0x02: Types table    │  ← 类型 ASM 文本池
│   Section 0x03: Attrs table    │  ← 属性字典池
│   Section 0x04: ConstPool      │  ← 常量池
│   Section 0x05: OpcodeSchema   │  ← op schema 扩展
│   Section 0x06: Module         │  ← op 字节码序列（紧凑 opcode）
│   Section 0x07: DebugInfo      │  ← 调试信息（可选）
│   Section 0x7F: Extra          │  ← 扩展数据
└────────────────────────────────┘
```

定义在 `tools/ptobc/include/ptobc/ptobc_format.h`。

### 编码策略

- **字符串去重**：所有类型名、属性名、op 名通过 StringTable 去重，引用时用 LEB128 编码的 ID
- **Opcode 紧凑编码**：每个 PTO op 有一个 uint8 opcode（定义在 `tools/ptobc/generated/ptobc_opcodes_v0.h`），不再存储完整的 `"pto.tmatmul"` 字符串
- **类型/属性引用**：类型和属性序列化为文本后存入池，通过 ID 引用
- **整数 LEB128 编码**：所有整数用 LEB128 可变长编码，小数字更紧凑

### 操作码变体

`.ptobc` 为 op 变体定义了紧凑编码：

| 变体 ID | 含义 |
|---|---|
| `kVariantDefault = 0` | 基础形式 |
| `kVariantAcc = 1` | 累加变体（如 `tmatmul.acc`） |
| `kVariantBias = 2` | 偏置变体（如 `tmatmul.bias`） |
| `kVariantMx = 3` | MX 混合精度变体 |
| `kVariantMxAcc = 4` | MX 累加变体 |
| `kVariantMxBias = 5` | MX 偏置变体 |

---

## 使用场景

### 各场景的实际使用

| 场景 | 流程 | 为什么 |
|---|---|---|
| **开发调试** | `.py → .pto → ptoas → .cpp` | 需要阅读和编辑 IR，`.pto` 人类可读 |
| **Lit 测试** | `.pto → ptoas → FileCheck` | 测试文本中的 CHECK 指令 |
| **CI 样例测试** | `.py → .pto → ptoas → .cpp` | 验证端到端正确性 |
| **Bytecode roundtrip 测试** | `.py → .pto → ptobc → .ptobc → .pto → ptoas → .cpp` | 验证编解码保真 |
| **生产部署**（预期） | 框架 → `.ptobc` → ptoas → .cpp | 紧凑存储 + 网络传输 |

### .pto 适用场景

- **开发调试**：直接阅读、编辑、git diff
- **Lit 回归测试**：`// RUN: ptoas %s | FileCheck %s` 需要 CHECK 指令匹配文本
- **Python 输出**：`python3 tmatmulk.py > tmatmulk.pto`
- **IR 中间查看**：`ptoas input.pto --emit-pto-ir` 输出经过部分 pass 的 IR

### .ptobc 适用场景

- **存储/传输**：比文本小 5-10 倍，节省空间和带宽
- **序列化部署**：AI 框架生成的 PTO IR 以紧凑格式存储和分发
- **Roundtrip 测试**：验证编解码的正确性
- **防篡改**：二进制格式不易被意外修改

### CI 中的 Roundtrip 测试

`--enablebc` 标志触发 bytecode roundtrip 测试（`test/samples/runop.sh`）：

```bash
bash test/samples/runop.sh --enablebc all
# 流程：.py → .pto → ptobc encode → .ptobc → ptobc decode → .pto → ptoas → .cpp
```

这验证了 `.ptobc` 编解码是保真的——经过编解码的 IR 与原始 IR 编译出相同的 C++。

部分 op（如 `alloc_tile` 的 addr 操作数、`tcvt` 的新形式）尚未被 ptobc v0 支持，会在 runop.sh 中自动跳过 roundtrip。

---

## 当前典型流程

```
Python (框架) ──构造IR──→ 内存中的 ModuleOp
                            │
                            │ print(module)  ← 唯一的输出方式
                            ▼
                         .pto 文本（磁盘文件）
                            │
                            ├──→ ptoas ──→ .cpp         ← 开发/调试
                            │
                            └──→ ptobc encode ──→ .ptobc ← 存储/部署
                                                      │
                                                  ptoas ──→ .cpp
```

**`.pto` 是当前必经的中间步骤**，因为 Python 绑定只提供 `print(module)` 输出文本格式。当前没有 Python API 可以直接生成 `.ptobc`。

---

## 生产环境的预期流程

从 README 的定位：

> PyPTO、TileLang、CuTile 等框架在 Python 端直接构建、操作和编译 PTO Bytecode

**预期**的生产流程：

```
AI 框架 (Python)
  │  构造 PTO IR (内存中)
  ▼
.ptobc (紧凑二进制)
  │  存储 / 版本管理 / 网络传输
  ▼
ptoas (目标机器)
  │  自动检测 "PTOBC\0" 魔术字节并解码
  ▼
.cpp kernel
  │
  ▼
昇腾硬件编译器 → NPU 执行
```

要达到这个流程，需要以下其中之一：

1. 给 Python 绑定添加 `ptobc_encode(module)` API（目前不存在）
2. 用管线 `print(module) → ptobc encode → .ptobc`（多一步但可行）

---

## 主流生态对比

| 项目 | 文本格式 | 二进制格式 | 关系 |
|---|---|---|---|
| **MLIR 上游** | `.mlir` 文本 | MLIR Bytecode（内置） | 同一 IR 的两种序列化 |
| **PTOAS** | `.pto` 文本 | `.ptobc` 自定义二进制 | 同一 IR 的两种序列化 |
| **LLVM IR** | `.ll` 文本 | `.bc` bitcode | 同一 IR 的两种序列化 |
| **TensorFlow** | `text_format` | `protobuf binary` | 同一计算图的两种序列化 |
| **ONNX** | JSON/text | protobuf binary | 同一模型的两种序列化 |

PTOAS 的 `.ptobc` 类似于 LLVM 的 `.bc` bitcode 或 MLIR 的 bytecode——都是同一个 IR 的紧凑二进制序列化，用于存储和传输；文本格式用于开发和调试。

---

## 总结

| 阶段 | 主要格式 | 原因 |
|---|---|---|
| **开发** | `.pto` | 可读、可编辑、可 diff |
| **测试** | `.pto`（主）+ `.ptobc`（roundtrip） | 验证两种格式等价 |
| **生产部署** | `.ptobc`（预期） | 紧凑、防篡改、适合传输 |

两者完全等价，`ptoas` 都能直接消费，选择取决于场景。
