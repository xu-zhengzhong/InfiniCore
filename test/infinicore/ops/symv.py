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
    # uplo, n, a_stride, x_stride, y_stride
    (0, 1, None, None, None),
    (0, 5, None, None, None),
    (0, 5, (5, 1), None, None),
    (0, 17, None, (2,), None),
    (0, 33, (1, 40), None, (2,)),
    (0, 33, (40, 1), None, (2,)),
    (0, 128, None, (2,), (3,)),
    (0, 1024, None, None, None),
    (1, 1, None, None, None),
    (1, 5, None, None, None),
    (1, 5, (5, 1), None, None),
    (1, 17, None, None, (2,)),
    (1, 33, (1, 40), (2,), None),
    (1, 33, (40, 1), (2,), None),
    (1, 128, None, (3,), (2,)),
    (1, 1024, None, None, None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _full_from_triangle(a, uplo):
    if uplo == 0:
        return torch.triu(a) + torch.triu(a, diagonal=1).t()
    return torch.tril(a) + torch.tril(a, diagonal=-1).t()


def torch_symv(alpha, a, x, beta, out, *, uplo=0):
    matrix = _full_from_triangle(a, uplo)
    result = alpha * torch.mv(matrix, x) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, a_stride, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            a_spec = TensorSpec.from_tensor(
                (n, n), a_stride if a_stride is not None else (1, n), dtype
            )
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            beta_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            y_spec = TensorSpec.from_tensor(
                (n,), y_stride, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, a_spec, x_spec, beta_spec, y_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=4,
                    tolerance=tol,
                    description="symv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 symv operator test"""

    def __init__(self):
        super().__init__("Symv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_symv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.symv(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
