import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase


def _summarize_sparse_data(rows, cols, crow, col, limit=6):
    nnz = len(col)
    density = nnz / (rows * cols) if rows and cols else 0

    def preview(values):
        if len(values) <= limit:
            return str(values)
        head = ", ".join(str(v) for v in values[: limit // 2])
        tail = ", ".join(str(v) for v in values[-(limit // 2) :])
        return f"[{head}, ..., {tail}]"

    return [
        f"rows={rows}",
        f"cols={cols}",
        f"nnz={nnz}",
        f"density={density:.6f}",
        f"crow={preview(crow)}",
        f"col={preview(col)}",
    ]


class SparseTestCase(TestCase):
    def __str__(self):
        input_str = "; ".join(str(inp) for inp in self.inputs)
        kwargs_strs = _summarize_sparse_data(
            self.kwargs["rows"],
            self.kwargs["cols"],
            self.kwargs["crow"],
            self.kwargs["col"],
        )
        out = self.kwargs.get("out")
        if out is not None:
            kwargs_strs.append(f"out={out}")
        return (
            f"TestCase({self.description} - inputs=[{input_str}], "
            f"kwargs={{{'; '.join(kwargs_strs)}}})"
        )


def _generate_spmv_cases():
    cases = []
    random.seed(42)
    # (rows, cols, density)
    configs = [
        (3, 4, 0.5),            # Baseline
        (1024, 1024, 0.02),     # 1K scale
        (4096, 4096, 0.005),    # 4K scale
    ]
    for rows, cols, density in configs:
        crow = [0]
        col = []
        for _ in range(rows):
            nnz_row = int(cols * density)
            if nnz_row > 0:
                col.extend(sorted(random.sample(range(cols), nnz_row)))
            crow.append(len(col))
        cases.append((rows, cols, crow, col))
    return cases

_TEST_CASES_DATA = _generate_spmv_cases()

# _TEST_CASES_DATA = [
#     (3, 4, [0, 2, 3, 5], [0, 2, 1, 0, 3]),
#     (4, 5, [0, 1, 1, 3, 4], [2, 0, 4, 1]),
# ]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [
    # infinicore.float16,
    # infinicore.bfloat16,
    infinicore.float32,
]


def parse_test_cases():
    test_cases = []
    for rows, cols, crow, col in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
            test_cases.append(
                SparseTestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((cols,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "crow": crow,
                        "col": col,
                    },
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpMV - OUT_OF_PLACE",
                )
            )
            test_cases.append(
                SparseTestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((cols,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "crow": crow,
                        "col": col,
                        "out": TensorSpec.from_tensor((rows,), dtype=dtype, name="out"),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpMV - OUT(out)",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpMV")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, values, x, *, rows, cols, crow, col, out=None):
        sparse = torch.sparse_csr_tensor(
            torch.tensor(crow, dtype=torch.int64, device=values.device),
            torch.tensor(col, dtype=torch.int64, device=values.device),
            values,
            size=(rows, cols),
        )
        result = torch.matmul(sparse, x)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, values, x, *, rows, cols, crow, col, out=None):
        device = values.device
        crow_tensor = infinicore.from_list(crow, dtype=infinicore.int64, device=device)
        col_tensor = infinicore.from_list(col, dtype=infinicore.int64, device=device)
        sparse = infinicore.csr_spmat(crow_tensor, col_tensor, values, (rows, cols))
        return infinicore.spmv(sparse, x, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
