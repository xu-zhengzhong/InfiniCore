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

_SIDE_UPLO_CASES = [(0, 0)]

_TEST_CASES_DATA = [
    # m, n
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
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


def _default_col_major_stride(rows):
    return (1, rows)


def parse_test_cases():
    test_cases = []
    for m, n in _TEST_CASES_DATA:
        a_stride_left = None
        a_stride_right = None
        b_stride = None
        c_stride = None
        for side, uplo in _SIDE_UPLO_CASES:
            for dtype in _TENSOR_DTYPES:
                dim_a = m if side == 0 else n
                a_stride = a_stride_left if side == 0 else a_stride_right
                if a_stride is None:
                    a_stride = _default_col_major_stride(dim_a)
                if b_stride is None:
                    b_stride = _default_col_major_stride(m)
                if c_stride is None:
                    c_stride = _default_col_major_stride(m)
                tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

                a_spec = TensorSpec.from_tensor((dim_a, dim_a), a_stride, dtype)
                b_spec = TensorSpec.from_tensor((m, n), b_stride, dtype)
                alpha_spec = TensorSpec.from_tensor(
                    (), None, dtype, init_mode=TensorInitializer.ONES
                )
                beta_spec = TensorSpec.from_tensor(
                    (), None, dtype, init_mode=TensorInitializer.ZEROS
                )
                c_spec = TensorSpec.from_tensor(
                    (m, n), c_stride, dtype, init_mode=TensorInitializer.RANDOM
                )

                kwargs = {"side": side, "uplo": uplo}

                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec, alpha_spec, beta_spec, c_spec],
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

    # def torch_operator(self, *args, **kwargs):
    #     return torch_symm(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.symm(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
