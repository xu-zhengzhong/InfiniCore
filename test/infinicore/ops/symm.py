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

_SIDE_UPLO_CASES = [(side, uplo) for side in (0, 1) for uplo in (0, 1)]

_TEST_CASES_DATA = [
    # m, n, a_stride_left, a_stride_right, b_stride, c_stride
    (1, 1, None, None, None, None),
    (1, 2, None, None, None, None),
    (1, 7, None, None, None, None),
    (2, 2, None, None, None, None),
    (3, 5, None, None, None, None),
    (5, 3, None, None, None, None),
    (8, 8, None, None, None, None),
    (9, 17, None, None, None, None),
    (17, 9, None, None, None, None),
    (31, 32, None, None, None, None),
    (32, 31, None, None, None, None),
    (32, 32, None, None, None, None),
    (33, 64, None, None, None, None),
    (64, 33, None, None, None, None),
    (65, 65, None, None, None, None),
    (127, 128, None, None, None, None),
    (128, 127, None, None, None, None),
    (256, 256, None, None, None, None),
    (512, 512, None, None, None, None),
    (1024, 1024, None, None, None, None),
    (1, 4096, None, None, None, None),
    (4096, 3, None, None, None, None),
    (3, 4096, None, None, None, None),
    (1, 4097, None, None, None, None),
    (17, 9, (1, 24), (1, 16), None, None),
    (17, 9, (24, 1), (16, 1), (12, 1), (13, 1)),
    (31, 32, (1, 40), (1, 48), (1, 36), (1, 40)),
    (32, 31, (40, 1), (48, 1), (35, 1), (37, 1)),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-4},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _full_from_triangle(a, uplo):
    if uplo == 0:
        return torch.triu(a) + torch.triu(a, diagonal=1).t()
    return torch.tril(a) + torch.tril(a, diagonal=-1).t()


def torch_symm(a, b, alpha, beta, out, *, side=0, uplo=0):
    matrix = _full_from_triangle(a, uplo)
    product = torch.mm(matrix, b) if side == 0 else torch.mm(b, matrix)
    result = alpha * product + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for m, n, a_stride_left, a_stride_right, b_stride, c_stride in _TEST_CASES_DATA:
        for side, uplo in _SIDE_UPLO_CASES:
            for dtype in _TENSOR_DTYPES:
                dim_a = m if side == 0 else n
                a_stride = a_stride_left if side == 0 else a_stride_right
                tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

                a_spec = TensorSpec.from_tensor((dim_a, dim_a), a_stride, dtype)
                b_spec = TensorSpec.from_tensor((m, n), b_stride, dtype)
                alpha_spec = TensorSpec.from_tensor(
                    (), None, dtype, init_mode=TensorInitializer.ONES
                )
                beta_one_spec = TensorSpec.from_tensor(
                    (), None, dtype, init_mode=TensorInitializer.ONES
                )
                c_spec = TensorSpec.from_tensor(
                    (m, n), c_stride, dtype, init_mode=TensorInitializer.RANDOM
                )

                kwargs = {"side": side, "uplo": uplo}

                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec, alpha_spec, beta_one_spec, c_spec],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=4,
                        tolerance=tol,
                        description="symm - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-3 symm operator test"""

    def __init__(self):
        super().__init__("Symm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_symm(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.symm(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
