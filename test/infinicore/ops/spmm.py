import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)

_TEST_CASES_DATA = [
    (3, 4, 2, [0, 2, 3, 5], [0, 2, 1, 0, 3]),
    (4, 5, 3, [0, 1, 1, 3, 4], [2, 0, 4, 1]),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# Sparse CSR tensor support is in beta state, so we only test float32 for now.
_TENSOR_DTYPES = [
    # infinicore.float16, 
    # infinicore.bfloat16, 
    infinicore.float32
]


def parse_test_cases():
    test_cases = []
    for rows, cols, n, crow, col in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((cols, n), dtype=dtype, name="b"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "crow": crow,
                        "col": col,
                    },
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpMM - OUT_OF_PLACE",
                )
            )
            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((cols, n), dtype=dtype, name="b"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "crow": crow,
                        "col": col,
                        "out": TensorSpec.from_tensor(
                            (rows, n), dtype=dtype, name="out"
                        ),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpMM - OUT(out)",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpMM")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, values, b, *, rows, cols, crow, col, out=None):
        sparse = torch.sparse_csr_tensor(
            torch.tensor(crow, dtype=torch.int64, device=values.device),
            torch.tensor(col, dtype=torch.int64, device=values.device),
            values,
            size=(rows, cols),
        )
        result = torch.matmul(sparse, b)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, values, b, *, rows, cols, crow, col, out=None):
        device = values.device
        crow_tensor = infinicore.from_list(crow, dtype=infinicore.int64, device=device)
        col_tensor = infinicore.from_list(col, dtype=infinicore.int64, device=device)
        sparse = infinicore.csr_spmat(crow_tensor, col_tensor, values, (rows, cols))
        return infinicore.spmm(sparse, b, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
    