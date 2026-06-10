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
from framework.tensor import TensorInitializer

import infinicore

_TEST_CASES_DATA = [
    # uplo, n, x_stride, y_stride
    (0, 128, (2,), (3,)),
    (0, 1024, None, None),
    (0, 4096, (2,), None),
    (0, 5120, None, (2,)),
    (1, 128, (3,), (2,)),
    (1, 1024, None, None),
    (1, 4096, None, (2,)),
    (1, 5120, (2,), None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _packed_to_full(ap, uplo, n):
    matrix = torch.empty((n, n), dtype=ap.dtype, device=ap.device)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=ap.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=ap.device)
    matrix[cols, rows] = ap
    matrix[rows, cols] = ap
    return matrix


def torch_blas_spmv(alpha, ap, x, beta, out, *, uplo=0):
    matrix = _packed_to_full(ap, uplo, x.shape[0])
    result = alpha * torch.mv(matrix, x) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            packed_len = n * (n + 1) // 2

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            ap_spec = TensorSpec.from_tensor((packed_len,), None, dtype)
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            beta_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            y_spec = TensorSpec.from_tensor(
                (n,), y_stride, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, ap_spec, x_spec, beta_spec, y_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=4,
                    tolerance=tol,
                    description="blas_spmv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 blas_spmv operator test"""

    def __init__(self):
        super().__init__("BlasSpmv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_blas_spmv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.blas_spmv(*args, **kwargs)


def main():
    torch.manual_seed(0)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
