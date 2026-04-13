import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
)

# Test cases format: (shape, x_stride)
# BLAS asum computes the sum of the absolute values of a 1-D tensor.
_TEST_CASES_DATA = [
    ((13,), None),
    ((13,), (2,)),
    ((5632,), None),
    ((5632,), (1,)),
    ((1024,), (4,)),
    ((2048,), (32,)),
]

# sum calculations can accumulate small floating point errors
_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.float64: {"atol": 1e-7, "rtol": 1e-5},
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
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            x_spec = TensorSpec.from_tensor(shape, strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"Asum - Shape {shape} Dtype {dtype}",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Asum operator test (Sum of absolute values)"""

    def __init__(self):
        super().__init__("Asum")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """Reference implementation using PyTorch"""
        x = args[0]
        # BLAS ASUM is equivalent to the L1 norm
        return torch.norm(x, p=1)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore implementation for BLAS asum."""
        return infinicore.asum(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()