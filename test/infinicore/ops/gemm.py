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
    (1, 1, 1, None, None, None),
    (1, 2, 1, None, None, None),
    (1, 7, 3, None, None, None),
    (2, 2, 2, None, None, None),
    (3, 5, 4, None, None, None),
    (5, 3, 4, None, None, None),
    (8, 8, 8, None, None, None),
    (9, 17, 11, None, None, None),
    (17, 9, 11, None, None, None),
    (31, 32, 16, None, None, None),
    (32, 31, 16, None, None, None),
    (32, 32, 32, None, None, None),
    (33, 64, 17, None, None, None),
    (64, 33, 17, None, None, None),
    (65, 65, 33, None, None, None),
    (127, 128, 64, None, None, None),
    (128, 127, 64, None, None, None),
    (256, 256, 128, None, None, None),
    (512, 512, 128, None, None, None),
    (1024, 1024, 256, None, None, None),
    (1, 4096, 16, None, None, None),
    (4096, 3, 16, None, None, None),
    (3, 4096, 16, None, None, None),
    (1, 4097, 16, None, None, None),
    (17, 9, 11, (1, 24), (1, 16), None),
    (17, 9, 11, (24, 1), (16, 1), (20, 1)),
    (31, 32, 16, (1, 40), (1, 48), (1, 40)),
    (32, 31, 16, (40, 1), (48, 1), (37, 1)),
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
