import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

_TEST_CASES_DATA = [
    # rows, cols, k, crow, col, alpha, beta
    (3, 4, 2, [0, 2, 3, 5], [0, 2, 1, 0, 3], 1.0, 0.0),
    (4, 5, 3, [0, 1, 1, 3, 4], [2, 0, 4, 1], 0.5, 1.0),
]

_TENSOR_DTYPES = [infinicore.float32]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def sampled_mm(a, b, values, rows, crow, col, alpha, beta):
    mm = torch.matmul(a, b)
    result = values.clone()
    for row in range(rows):
        for ptr in range(crow[row], crow[row + 1]):
            result[ptr] = alpha * mm[row, col[ptr]] + beta * values[ptr]
    return result


class SparseTestCase(TestCase):
    def __str__(self):
        nnz = len(self.kwargs["col"])
        return (
            f"TestCase({self.description} - "
            f"rows={self.kwargs['rows']}; cols={self.kwargs['cols']}; "
            f"k={self.kwargs['k']}; nnz={nnz}; alpha={self.kwargs['alpha']}; "
            f"beta={self.kwargs['beta']})"
        )


def parse_test_cases():
    test_cases = []
    for rows, cols, k, crow, col, alpha, beta in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
            test_cases.append(
                SparseTestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((rows, k), dtype=dtype, name="a"),
                        TensorSpec.from_tensor((k, cols), dtype=dtype, name="b"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "k": k,
                        "crow": crow,
                        "col": col,
                        "alpha": alpha,
                        "beta": beta,
                    },
                    comparison_target=0,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SDDMM - INPLACE",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SDDMM")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, values, a, b, *, rows, cols, k, crow, col, alpha, beta):
        del cols, k
        values.copy_(sampled_mm(a, b, values, rows, crow, col, alpha, beta))
        return values

    def infinicore_operator(self, values, a, b, *, rows, cols, k, crow, col, alpha, beta):
        del k
        device = values.device
        crow_tensor = infinicore.from_list(crow, dtype=infinicore.int64, device=device)
        col_tensor = infinicore.from_list(col, dtype=infinicore.int64, device=device)
        sparse = infinicore.csr_spmat(crow_tensor, col_tensor, values, (rows, cols))
        infinicore.sddmm(sparse, a, b, alpha=alpha, beta=beta)
        return values


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
