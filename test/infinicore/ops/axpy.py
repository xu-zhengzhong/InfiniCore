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

# Test cases format: (shape, x_strides, y_strides)

_TEST_CASES_DATA = [
    ((8,), None, None),
    ((8,), (16,), (16,)),
    ((24,), None, None),
    ((5632,), None, None),
    ((5632,), (5,), (5,)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.float64: {"atol": 1e-7, "rtol": 1e-6},
}

_TENSOR_DTYPES = [
    # infinicore.float16,
    infinicore.float32,
    infinicore.float64,
]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, x_strides, y_strides = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)
            alpha_spec = TensorSpec.from_tensor((), None, dtype)

            test_cases.append(
                TestCase(
                    inputs=[y_spec, x_spec, alpha_spec],
                    kwargs=None,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"Axpy - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Axpy operator test (y = alpha * x + y)"""

    def __init__(self):
        super().__init__("Axpy")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch axpy implementation"""
        y = args[0]
        x = args[1]
        alpha = args[2]
        # return torch.add(torch.mul(x, alpha), y)
        return torch.add(y, x, alpha=alpha.item())
        # return torch.add(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore axpy implementation"""
        return infinicore.axpy(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()