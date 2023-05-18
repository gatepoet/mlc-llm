# pylint: disable=missing-docstring,invalid-name
import argparse
import os
import sys
from typing import Callable, List, Optional

from tvm import meta_schedule as ms
from tvm import tir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="dist/dolly-v2-3b/float16/",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="dist/dolly-v2-3b/float16/log_db/",
    )
    parser.add_argument("--target", type=str, default="auto")
    args = parser.parse_args()
    sys.path.insert(0, f"{args.path}/debug")
    from mod_tir_static import (  # type: ignore  # pylint: disable=import-outside-toplevel,import-error
        Module,
    )

    from mlc_llm import utils  # pylint: disable=import-outside-toplevel

    args.mod = Module
    utils.parse_target(args)
    os.makedirs(args.work_dir, exist_ok=True)
    print("target:", args.target)
    return args


def sch_matmul() -> Callable:
    def sch_func(sch: tir.Schedule) -> None:
        b0 = sch.get_block(name="matmul", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.pad_einsum(b0, [1, 64, 64])
        (b2,) = sch.get_producers(b0)
        (b3,) = sch.get_consumers(b0)
        sch.annotate(
            block_or_loop=b0,
            ann_key="meta_schedule.tiling_structure",
            ann_val="SSSRRSRS",
        )
        l2, l3, l4 = sch.get_loops(block=b0)
        l10, l11, l12, l13, l14 = sch.split(
            loop=l2, factors=[1, 1, 1, 1, 1], preserve_unit_iters=True
        )
        l20, l21, l22, l23, l24 = sch.split(
            loop=l3, factors=[393, 1, 64, 1, 2], preserve_unit_iters=True
        )
        v25, v26, v27 = sch.sample_perfect_tile(
            loop=l4, n=3, max_innermost_factor=64, decision=[128, 2, 8]
        )
        l28, l29, l30 = sch.split(
            loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True
        )
        sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
        l31 = sch.fuse(l10, l20, preserve_unit_iters=True)
        sch.bind(loop=l31, thread_axis="blockIdx.x")
        l32 = sch.fuse(l11, l21, preserve_unit_iters=True)
        sch.bind(loop=l32, thread_axis="vthread.x")
        l33 = sch.fuse(l12, l22, preserve_unit_iters=True)
        sch.bind(loop=l33, thread_axis="threadIdx.x")
        sch.annotate(
            block_or_loop=b0,
            ann_key="meta_schedule.thread_extent_low_inclusive",
            ann_val=32,
        )
        sch.annotate(
            block_or_loop=b0,
            ann_key="meta_schedule.thread_extent_high_inclusive",
            ann_val=1024,
        )
        b34 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
        sch.reverse_compute_at(block=b34, loop=l33, preserve_unit_loops=True, index=-1)
        b35 = sch.cache_read(
            block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0]
        )
        sch.compute_at(block=b35, loop=l28, preserve_unit_loops=True, index=-1)
        l36, l37, l38, l39, l40, l41 = sch.get_loops(block=b35)
        l42 = sch.fuse(l40, l41, preserve_unit_iters=True)
        v43 = sch.sample_categorical(
            candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25]
        )
        sch.annotate(
            block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch", ann_val=v43
        )
        b44 = sch.cache_read(
            block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0]
        )
        sch.compute_at(block=b44, loop=l28, preserve_unit_loops=True, index=-1)
        l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b44)
        l51 = sch.fuse(l49, l50, preserve_unit_iters=True)
        v52 = sch.sample_categorical(
            candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25]
        )
        sch.annotate(
            block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v52
        )
        v53 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024], probs=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        sch.annotate(
            block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v53
        )
        # sch.enter_postproc()

        sch.unannotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch")
        l54, l55, l56, l57, l58 = sch.get_loops(block=b35)
        l59, l60, l61 = sch.split(
            loop=l58, factors=[None, 64, 2], preserve_unit_iters=True
        )
        sch.vectorize(loop=l61)
        sch.bind(loop=l60, thread_axis="threadIdx.x")
        sch.unannotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch")
        l62, l63, l64, l65, l66 = sch.get_loops(block=b44)
        l67, l68, l69 = sch.split(
            loop=l66, factors=[None, 64, 2], preserve_unit_iters=True
        )
        sch.vectorize(loop=l69)
        sch.bind(loop=l68, thread_axis="threadIdx.x")
        b70 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b70, ann_key="meta_schedule.unroll_explicit")
        b104 = sch.get_block(name="matmul", func_name="main")
        l105, l106, l107, l108, l109, l110, l111, l112, l113, l114 = sch.get_loops(
            block=b104
        )
        b115 = sch.decompose_reduction(block=b104, loop=l108)

        sch.compute_inline(b2)
        sch.reverse_compute_inline(b3)

    return sch_func


def sch_fused_decode_gemv(
    name_gemv: str = "matmul",
    name_decode: str = "decode",
    name_epilogues: Optional[List[str]] = None,
) -> Callable:
    def sch_func(sch: tir.Schedule) -> None:
        gemv = sch.get_block(name_gemv)
        decode = sch.get_block(name_decode)
        epilogues = (
            [sch.get_block(n) for n in name_epilogues]
            if name_epilogues is not None
            else []
        )
        # Step 1. Schedule GEMV
        # [b=1, i=1, j, k]
        # split j           => [b=1, i=1, (bx, vx, tx), k]
        # fuse (b, i, bx)   => [bx, vx, tx, (k)]
        # split k           => [bx, vx, tx, (ko, ki * 8)]
        rb = sch.cache_write(gemv, 0, "local")
        b, i, j, k = sch.get_loops(gemv)
        assert sch.get(b).extent.value == 1
        assert sch.get(i).extent.value == 1
        len_bx, len_vx, len_tx = sch.sample_perfect_tile(
            j, n=3, max_innermost_factor=-1
        )
        bx, vx, tx = sch.split(j, [len_bx, len_vx, len_tx])
        bx = sch.fuse(b, i, bx)
        k_o, k_8 = sch.split(k, [None, 8])
        k_o, k_i = sch.split(
            k_o,
            sch.sample_perfect_tile(k_o, n=2, max_innermost_factor=-1),
        )
        k_i = sch.fuse(k_i, k_8)
        sch.unroll(k_i)
        sch.bind(bx, thread_axis="blockIdx.x")
        sch.bind(vx, thread_axis="vthread.x")
        sch.bind(tx, thread_axis="threadIdx.x")
        sch.reorder(bx, vx, tx, k_o, k_i)
        # Step 2. Schedule decode: move to under threadIdx.x and fetch separately for each thread
        sch.compute_at(decode, k_o, preserve_unit_loops=True)
        sch.set_scope(decode, 0, "local")
        _, unroll = sch.split(sch.get_loops(decode)[-2], [None, 8])
        sch.unroll(unroll)

        # Step 3. Cooperative fetch GEMV
        def cooperative_fetch(block, tx):
            block = sch.cache_read(block, 0, "shared")
            sch.compute_at(block, tx, preserve_unit_loops=True)
            l = sch.fuse(*sch.get_loops(block)[-2:])
            len_vector = sch.sample_categorical(
                [1, 2, 3, 4],
                probs=[0.25, 0.25, 0.25, 0.25],
            )
            _, tx, vec = sch.split(l, [None, len_tx, len_vector])
            sch.bind(tx, thread_axis="threadIdx.x")
            sch.vectorize(vec)
            sch.storage_align(block, buffer_index=0, axis=-2, factor=32, offset=8)

        cooperative_fetch(gemv, tx)
        # Step 4. Schedule epilogue
        for epilogue in epilogues[:-1]:
            sch.compute_inline(epilogue)
        if epilogues:
            sch.reverse_compute_inline(epilogues[-1])
        sch.reverse_compute_at(rb, tx, preserve_unit_loops=True)
        # Step 5. Postprocess: decompose reduction
        sch.decompose_reduction(gemv, k_o)
        sch.show(black_format=False)

    return sch_func


def sch_decode(
    name_decode: str = "decode",
    name_transpose: Optional[str] = "T_transpose",
):
    def sch_func(sch: tir.Schedule) -> None:
        decode = sch.get_block(name_decode)
        # Step 1. Tile the decoding
        i, j = sch.get_loops(decode)
        i, i_8 = sch.split(i, factors=[None, 8])
        len_by, len_ty, len_i = sch.sample_perfect_tile(i, n=3, max_innermost_factor=4)
        len_bx, len_tx = sch.sample_perfect_tile(j, n=2, max_innermost_factor=16)
        by, ty, i = sch.split(i, [len_by, len_ty, len_i])
        bx, tx = sch.split(j, [len_bx, len_tx])
        sch.reorder(by, bx, ty, tx, i, i_8)
        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.unroll(i_8)
        # Step 2. Cache results in shared memory
        rb = sch.cache_write(decode, 0, storage_scope="shared")
        if name_transpose is None:
            epilogue = rb
        else:
            sch.compute_inline(rb)
            epilogue = sch.get_block(name_transpose)
        # Step 3. Schedule the shared memory write back
        sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
        l = sch.fuse(*sch.get_loops(epilogue)[-2:])
        _, ty, tx, vec = sch.split(l, factors=[None, len_ty, len_tx, 4])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)
        sch.storage_align(decode, buffer_index=0, axis=0, factor=32, offset=1)
        sch.mod.show(black_format=False)

    return sch_func


def main():
    args = _parse_args()
    path_workload = os.path.join(args.path, "database_workload.json")
    path_tuning_record = os.path.join(args.path, "database_tuning_record.json")
    if os.path.exists(path_workload):
        os.remove(path_workload)
    if os.path.exists(path_tuning_record):
        os.remove(path_tuning_record)

    args.mod.show(black_format=False)
    database = ms.database.MemoryDatabase()
    # runner = ms.runner.RPCRunner(
    #     ms.runner.RPCConfig(
    #         tracker_host="192.168.10.1",
    #         tracker_port=9191,
    #         tracker_key="m2-mac-mini",
    #     ),
    # )
    runner = ms.runner.LocalRunner(timeout_sec=10)
    ms.tir_integration.tune_tir(
        mod=args.mod,
        target=args.target,
        work_dir=args.work_dir,
        max_trials_global=2000,
        max_trials_per_task=1000,
        runner=runner,
        database=database,
        space="cuda",
        special_space={
            "fused_decode1_fused_matmul2_add1_gelu": sch_fused_decode_gemv(
                name_epilogues=[
                    "T_add",
                    "T_multiply",
                    "compute",
                    "compute_1",
                    "compute_2",
                    "T_multiply_1",
                    "T_add_1",
                    "T_multiply_2",
                ]
            ),
            "fused_decode2_fused_matmul3_add": sch_fused_decode_gemv(
                name_epilogues=["T_add"]
            ),
            "fused_decode3_matmul4": sch_fused_decode_gemv(),
            "fused_decode_fused_matmul_add": sch_fused_decode_gemv(
                name_epilogues=["T_add"]
            ),
            "decode": sch_decode(name_transpose=None),
            "decode4": sch_decode(),
            "decode5": sch_decode(),
            "decode6": sch_decode(),
            "matmul3": sch_matmul(),
        },
    )
    if os.path.exists(path_workload):
        os.remove(path_workload)
    if os.path.exists(path_tuning_record):
        os.remove(path_tuning_record)
    database.dump_pruned(ms.database.JSONDatabase(work_dir=args.path))


if __name__ == "__main__":
    main()
