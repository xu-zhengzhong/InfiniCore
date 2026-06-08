import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

_TEST_CASES_DATA = [
    ((3,), None, None, 1.0, 0.0),
    ((257,), (2,), None, 0.5, 1.0),
    ((4096,), None, (2,), -1.25, 0.25),
]

_TENSOR_DTYPES = [infinicore.float32]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def parse_test_cases():
    test_cases = []
    for shape, x_strides, y_strides, alpha, beta in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor(shape, x_strides, dtype, name="x"),
                        TensorSpec.from_tensor(shape, y_strides, dtype, name="y"),
                    ],
                    kwargs={"alpha": alpha, "beta": beta},
                    output_spec=None,
                    comparison_target=1,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="Axpby - INPLACE",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Axpby")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, y, *, alpha, beta):
        y.copy_(alpha * x + beta * y)
        return y

    def infinicore_operator(self, x, y, *, alpha, beta):
        return infinicore.axpby(x, y, alpha=alpha, beta=beta, out=y)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
