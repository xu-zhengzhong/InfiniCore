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
    is_broadcast,
)

# Test cases format: (x_shape, x_strides_or_None, alpha)
# scal computes y = alpha * x

_TEST_CASES_DATA = [
    ((8,), None, 2.0),
    ((8,), (16,), -0.5),
    ((24,), None, 1.5),
    ((5632,), None, 2.5),
    ((5632,), (5,), 2.5),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.float64: {"atol": 1e-7, "rtol": 1e-6},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}

_TENSOR_DTYPES = [
    infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    infinicore.bfloat16,
]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, alpha = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            x_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {"alpha": alpha}

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"Scal - OUT_OF_PLACE (alpha={alpha})",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Scal operator test (y = alpha * x)"""

    def __init__(self):
        super().__init__("Scal")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        alpha = kwargs.pop("alpha", 1.0)
        return torch.mul(args[0], alpha, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.scal(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()