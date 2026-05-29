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
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
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
    return matrix


def torch_tpmv(ap, x, *, uplo=0, trans=0, diag=0):
    matrix = _packed_to_triangular(ap, uplo, diag, x.shape[0])
    op_matrix = matrix if trans == 0 else matrix.t()
    result = torch.mv(op_matrix, x.clone())
    x.copy_(result)
    return x


def parse_test_cases():
    test_cases = []
    for uplo, trans, diag, n, x_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
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
                    description="tpmv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 tpmv operator test"""

    def __init__(self):
        super().__init__("Tpmv")

    def get_test_cases(self):
        return parse_test_cases()

    # def torch_operator(self, *args, **kwargs):
    #     return torch_tpmv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.tpmv(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
