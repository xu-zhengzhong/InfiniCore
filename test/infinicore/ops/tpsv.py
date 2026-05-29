import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)

import infinicore

_TEST_CASES_DATA = [
    # uplo, trans, diag, n, x_stride
    (0, 0, 0, n, None)
    for n in (4096, 6144, 8192)
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 5e-3, "rtol": 5e-3},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _packed_to_triangular(ap, uplo, diag, n):
    matrix = torch.zeros((n, n), dtype=ap.dtype, device=ap.device)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=ap.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=ap.device)
    matrix[cols, rows] = ap
    if diag == 1:
        matrix.diagonal().fill_(1)
    else:
        diag_values = matrix.diagonal()
        diag_values.copy_(
            diag_values.sign().masked_fill(diag_values == 0, 1)
            * (diag_values.abs() + 2)
        )
    return matrix


def _triangular_to_packed(matrix, uplo):
    n = matrix.shape[0]
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=matrix.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=matrix.device)
    return matrix[cols, rows].contiguous()


def torch_tpsv(ap, x, *, uplo=0, trans=0, diag=0):
    rhs = x.clone()
    matrix = _packed_to_triangular(ap, uplo, diag, x.shape[0])
    ap.copy_(_triangular_to_packed(matrix, uplo))
    op_matrix = matrix if trans == 0 else matrix.t()
    result = torch.linalg.solve_triangular(
        op_matrix,
        rhs.unsqueeze(1),
        upper=(uplo == 0 if trans == 0 else uplo == 1),
        unitriangular=(diag == 1),
    ).squeeze(1)
    x.copy_(result)
    return x


def parse_test_cases():
    test_cases = []
    for uplo, trans, diag, n, x_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 5e-4, "rtol": 5e-4})
            packed_len = n * (n + 1) // 2

            ap_spec = TensorSpec.from_tensor((packed_len,), None, dtype)
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)

            test_cases.append(
                TestCase(
                    inputs=[ap_spec, x_spec],
                    kwargs={"uplo": uplo, "trans": trans, "diag": diag},
                    output_spec=None,
                    comparison_target=1,
                    tolerance=tol,
                    description="tpsv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 tpsv operator test"""

    def __init__(self):
        super().__init__("Tpsv")

    def get_test_cases(self):
        return parse_test_cases()

    # def torch_operator(self, *args, **kwargs):
    #     return torch_tpsv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.tpsv(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
