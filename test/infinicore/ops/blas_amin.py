import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

# Test cases format: (shape, x_stride)
# BLAS amin computes the 1-based index of the minimum absolute value of a 1-D tensor.

_TEST_CASES_DATA = [
    ((13,), None),
    ((13,), (10,)),
    ((5632,), None),
    ((5632,), (5,)),
    ((16,), (4,)),
    ((5632,), (32,)),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 0.0, "rtol": 0.0},
    infinicore.float64: {"atol": 0.0, "rtol": 0.0},
}

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0.0, "rtol": 0.0})
            x_spec = TensorSpec.from_tensor(shape, strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="BlasAmin - 1D Index",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 amin operator test"""

    def __init__(self):
        super().__init__("BlasAmin")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        x = args[0]
        return torch.argmin(x.abs()) + 1

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.blas_amin(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
