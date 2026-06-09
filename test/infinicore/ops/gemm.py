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
    # m, n, k, a_stride, b_stride, c_stride
    (128, 128, 128, None, None, None),
    (1024, 1024, 1024, None, None, None),
    (4096, 4096, 4096, None, None, None),
    (5120, 5120, 5120, None, None, None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def torch_gemm(a, b, alpha, beta, out):
    result = alpha * torch.mm(a, b) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for m, n, k, a_stride, b_stride, c_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})

            a_spec = TensorSpec.from_tensor((m, k), a_stride, dtype)
            b_spec = TensorSpec.from_tensor((k, n), b_stride, dtype)
            c_spec = TensorSpec.from_tensor(
                (m, n), c_stride, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec, 1.0, 1.0, c_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=4,
                    tolerance=tol,
                    description="gemm - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-3 gemm operator test"""

    def __init__(self):
        super().__init__("Gemm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_gemm(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.gemm(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
