# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv3_f32 = pto.TensorViewType.get(3, f32, ctx)
            tile_view_3d = pto.PartitionTensorViewType.get([1, 32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("sigmoid_kernel_3d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c2 = arith.ConstantOp(IndexType.get(ctx), 2).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                c128 = arith.ConstantOp(IndexType.get(ctx), 128).result
                one = arith.ConstantOp(f32, 1.0).result

                arg0, arg1 = entry.arguments

                tv0 = pto.MakeTensorViewOp(tv3_f32, arg0, [c2, c128, c128], [arith.ConstantOp(IndexType.get(ctx), 16384).result, c128, c1]).result
                tv1 = pto.MakeTensorViewOp(tv3_f32, arg1, [c2, c128, c128], [arith.ConstantOp(IndexType.get(ctx), 16384).result, c128, c1]).result

                tb0 = pto.AllocTileOp(tile_buf_32).result
                tb1 = pto.AllocTileOp(tile_buf_32).result
                tb2 = pto.AllocTileOp(tile_buf_32).result
                tb3 = pto.AllocTileOp(tile_buf_32).result

                for i in range(0, 2, 1):
                    for j in range(0, 128, 32):
                        for k in range(0, 128, 32):
                            ci = arith.ConstantOp(IndexType.get(ctx), i).result
                            cj = arith.ConstantOp(IndexType.get(ctx), j).result
                            ck = arith.ConstantOp(IndexType.get(ctx), k).result

                            sv0 = pto.PartitionViewOp(tile_view_3d, tv0, offsets=[ci, cj, ck], sizes=[c1, c32, c32]).result
                            sv1 = pto.PartitionViewOp(tile_view_3d, tv1, offsets=[ci, cj, ck], sizes=[c1, c32, c32]).result

                            pto.TLoadOp(None, sv0, tb0)

                            pto.TNegOp(tb0, tb1)
                            pto.TExpOp(tb1, tb2)
                            pto.TAddSOp(tb2, one, tb3)
                            pto.TRecipOp(tb3, tb1)

                            pto.TStoreOp(None, tb1, sv1)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
