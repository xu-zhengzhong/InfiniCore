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
    # trans, m, n, a_stride, x_stride, y_stride
    (0, 128, 128, None, (2,), (3,)),
    (0, 1024, 1024, None, None, None),
    (0, 4096, 4096, None, (2,), None),
    (0, 5120, 5120, None, None, (2,)),
    (1, 128, 128, None, (2,), (3,)),
    (1, 1024, 1024, None, None, None),
    (1, 4096, 4096, None, None, (2,)),
    (1, 5120, 5120, None, (2,), None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def torch_gemv(alpha, a, x, beta, out, *, trans=0):
    matrix = a if trans == 0 else a.t()
    result = alpha * torch.mv(matrix, x) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for trans, m, n, a_stride, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x_len = n if trans == 0 else m
            y_len = m if trans == 0 else n

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            a_spec = TensorSpec.from_tensor((m, n), a_stride, dtype)
            x_spec = TensorSpec.from_tensor((x_len,), x_stride, dtype)
            beta_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            y_spec = TensorSpec.from_tensor(
                (y_len,), y_stride, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, a_spec, x_spec, beta_spec, y_spec],
                    kwargs={"trans": trans},
                    output_spec=None,
                    comparison_target=4,
                    tolerance=tol,
                    description="gemv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 gemv operator test"""

    def __init__(self):
        super().__init__("Gemv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_gemv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.gemv(*args, **kwargs)


def main():
    torch.manual_seed(0)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
