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
    # uplo, trans, diag, n, a_stride, x_stride
    (0, 0, 0, 128, None, (3,)),
    (0, 1, 0, 1024, None, None),
    (0, 0, 1, 4096, None, (2,)),
    (0, 1, 1, 5120, None, None),
    (1, 0, 0, 128, None, (3,)),
    (1, 1, 0, 1024, None, None),
    (1, 0, 1, 4096, None, None),
    (1, 1, 1, 5120, None, (2,)),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _triangular(a, uplo, diag):
    matrix = torch.triu(a) if uplo == 0 else torch.tril(a)
    if diag == 1:
        matrix = matrix.clone()
        matrix.diagonal().fill_(1)
    return matrix


def torch_trmv(a, x, *, uplo=0, trans=0, diag=0):
    matrix = _triangular(a, uplo, diag)
    op_matrix = matrix if trans == 0 else matrix.t()
    result = torch.mv(op_matrix, x.clone())
    x.copy_(result)
    return x


def parse_test_cases():
    test_cases = []
    for uplo, trans, diag, n, a_stride, x_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            a_spec = TensorSpec.from_tensor(
                (n, n), a_stride if a_stride is not None else (1, n), dtype
            )
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)

            test_cases.append(
                TestCase(
                    inputs=[a_spec, x_spec],
                    kwargs={"uplo": uplo, "trans": trans, "diag": diag},
                    output_spec=None,
                    comparison_target=1,
                    tolerance=tol,
                    description="trmv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 trmv operator test"""

    def __init__(self):
        super().__init__("Trmv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_trmv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.trmv(*args, **kwargs)


def main():
    torch.manual_seed(0)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
