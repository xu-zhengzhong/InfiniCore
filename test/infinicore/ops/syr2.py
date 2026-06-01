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
    (0, 5, (5, 1), None, (2,)),
    (0, 17, None, (2,), None),
    (0, 33, (1, 40), None, (2,)),
    (0, 33, (40, 1), (2,), None),
    (0, 128, None, (2,), (3,)),
    (0, 256, None, None, None),
    (1, 1, None, None, None),
    (1, 5, None, None, None),
    (1, 5, (5, 1), (2,), None),
    (1, 17, None, None, (2,)),
    (1, 33, (1, 40), (2,), None),
    (1, 33, (40, 1), None, (2,)),
    (1, 128, None, (3,), (2,)),
    (1, 256, None, None, None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _triangle_update(a, update, uplo):
    if uplo == 0:
        return torch.triu(a + update) + torch.tril(a, diagonal=-1)
    return torch.tril(a + update) + torch.triu(a, diagonal=1)


def torch_syr2(alpha, x, y, out, *, uplo=0):
    result = _triangle_update(
        out, alpha * (torch.outer(x, y) + torch.outer(y, x)), uplo
    )
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
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            y_spec = TensorSpec.from_tensor((n,), y_stride, dtype)
            a_spec = TensorSpec.from_tensor(
                (n, n),
                a_stride if a_stride is not None else (1, n),
                dtype,
                init_mode=TensorInitializer.RANDOM,
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, x_spec, y_spec, a_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=3,
                    tolerance=tol,
                    description="syr2 - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 syr2 operator test"""

    def __init__(self):
        super().__init__("Syr2")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_syr2(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.syr2(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
