import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

# Test cases format: (shape, x_strides, y_strides)

_TEST_CASES_DATA = [
    ((8,), None, None),
    ((8,), (16,), (16,)),
    ((24,), None, None),
    ((5632,), None, None),
    ((5632,), (5,), (5,)),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-7, "rtol": 1e-7},
    infinicore.float64: {"atol": 1e-15, "rtol": 1e-15},
}

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]


def parse_test_cases():
    test_cases = []
    for shape, x_strides, y_strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[x_spec, y_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=0,
                    tolerance=tol,
                    description="Copy - INPLACE(x)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 copy operator test"""

    def __init__(self):
        super().__init__("Copy")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        x, y = args
        x.copy_(y)
        return x

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.copy(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
