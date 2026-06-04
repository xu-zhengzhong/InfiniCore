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

_SIDE_UPLO_TRANS_DIAG_CASES = [
    (side, uplo, trans, diag)
    for side in (0, 1)
    for uplo in (0, 1)
    for trans in (0, 1)
    for diag in (0, 1)
]

_TEST_CASES_DATA = [
    # m, n, a_stride_left, a_stride_right, b_stride
    (1, 1, None, None, None),
    (1, 2, None, None, None),
    (2, 2, None, None, None),
    (3, 5, None, None, None),
    (5, 3, None, None, None),
    (8, 8, None, None, None),
    (17, 9, None, None, None),
    (31, 32, None, None, None),
    (32, 31, None, None, None),
    (65, 65, None, None, None),
    (127, 128, None, None, None),
    (128, 127, None, None, None),
    (256, 256, None, None, None),
    (512, 512, None, None, None),
    (1024, 1024, None, None, None),
    (1, 4096, None, None, None),
    (17, 9, (1, 24), (1, 16), None),
    (17, 9, (24, 1), (16, 1), (12, 1)),
    (31, 32, (1, 40), (1, 48), (1, 36)),
    (32, 31, (40, 1), (48, 1), (35, 1)),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 3e-3, "rtol": 1e-3},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _default_col_major_stride(rows):
    return (1, rows)


def _full_from_triangle(a, uplo, diag):
    triangular = torch.triu(a) if uplo == 0 else torch.tril(a)
    if diag == 1:
        triangular = triangular.clone()
        triangular.diagonal().fill_(1)
    return triangular


def _condition_triangular_input(a):
    a.diagonal().add_(2.0)


def torch_trsm(a, alpha, b, *, side=0, uplo=0, trans=0, diag=0):
    _condition_triangular_input(a)
    triangular = _full_from_triangle(a, uplo, diag)
    op_a = triangular if trans == 0 else triangular.t()
    upper = (uplo == 0) if trans == 0 else (uplo == 1)
    rhs = alpha * b

    if side == 0:
        result = torch.linalg.solve_triangular(
            op_a,
            rhs,
            upper=upper,
            left=True,
            unitriangular=diag == 1,
        )
    else:
        result = torch.linalg.solve_triangular(
            op_a.t(),
            rhs.t(),
            upper=not upper,
            left=True,
            unitriangular=diag == 1,
        ).t()

    b.copy_(result)
    return b


def parse_test_cases():
    test_cases = []
    for m, n, a_stride_left, a_stride_right, b_stride in _TEST_CASES_DATA:
        for side, uplo, trans, diag in _SIDE_UPLO_TRANS_DIAG_CASES:
            for dtype in _TENSOR_DTYPES:
                dim_a = m if side == 0 else n
                a_stride = a_stride_left if side == 0 else a_stride_right
                if a_stride is None:
                    a_stride = _default_col_major_stride(dim_a)
                if b_stride is None:
                    b_stride = _default_col_major_stride(m)
                tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

                a_spec = TensorSpec.from_tensor(
                    (dim_a, dim_a), a_stride, dtype, scale=0.2
                )
                alpha_spec = TensorSpec.from_tensor(
                    (), None, dtype, init_mode=TensorInitializer.ONES
                )
                b_spec = TensorSpec.from_tensor(
                    (m, n),
                    b_stride,
                    dtype,
                    init_mode=TensorInitializer.RANDOM,
                    scale=0.5,
                )

                kwargs = {"side": side, "uplo": uplo, "trans": trans, "diag": diag}

                test_cases.append(
                    TestCase(
                        inputs=[a_spec, alpha_spec, b_spec],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=2,
                        tolerance=tol,
                        description="trsm - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-3 trsm operator test"""

    def __init__(self):
        super().__init__("Trsm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_trsm(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.trsm(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
