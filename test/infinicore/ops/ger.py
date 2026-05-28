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
    # m, n, a_stride, x_stride, y_stride
    (1, 1, None, None, None),
    (3, 4, (1, 3), None, None),
    (4, 5, (8, 1), (2,), (3,)),
    (7, 3, (1, 9), (2,), (3,)),
    (32, 17, None, None, None),
    (64, 33, (1, 66), (2,), (2,)),
    (16, 5632, None, (2,), (2,)),
    (5632, 33, (1, 5632), None, None),
    (2048, 2560, (1, 4096), None, None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def torch_ger(alpha, x, y, out):
    out.add_(torch.outer(x, y), alpha=alpha.item())
    return out


def parse_test_cases():
    test_cases = []
    for m, n, a_stride, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            x_spec = TensorSpec.from_tensor((m,), x_stride, dtype)
            y_spec = TensorSpec.from_tensor((n,), y_stride, dtype)
            a_spec = TensorSpec.from_tensor(
                (m, n), a_stride, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, x_spec, y_spec, a_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=3,
                    tolerance=tol,
                    description="ger - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 ger operator test"""

    def __init__(self):
        super().__init__("Ger")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_ger(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.ger(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
