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

import infinicore

_TEST_CASES_DATA = [
    # uplo, trans, diag, n, k, a_stride, x_stride
    (0, 0, 0, n, 64, None, None)
    for n in (4096, 6144, 8192)
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 5e-4, "rtol": 5e-4},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _full_from_band(a, uplo, diag, n, k):
    full = torch.zeros((n, n), dtype=a.dtype, device=a.device)
    if uplo == 0:
        for j in range(n):
            i_begin = max(0, j - k)
            for i in range(i_begin, j + 1):
                full[i, j] = 1 if diag == 1 and i == j else a[k + i - j, j]
    else:
        for j in range(n):
            i_end = min(n, j + k + 1)
            for i in range(j, i_end):
                full[i, j] = 1 if diag == 1 and i == j else a[i - j, j]
    return full


def _stabilize_band(a, uplo, diag, k):
    band = a.clone()
    if diag == 0:
        diag_row = k if uplo == 0 else 0
        diag_values = band[diag_row]
        diag_values.copy_(
            diag_values.sign().masked_fill(diag_values == 0, 1)
            * (diag_values.abs() + 2)
        )
    return band


def torch_tbsv(a, x, *, uplo=0, trans=0, diag=0, k=0):
    x_input = x.clone()
    band = _stabilize_band(a, uplo, diag, k)
    a.copy_(band)
    matrix = _full_from_band(a, uplo, diag, x.shape[0], k)
    op_matrix = matrix if trans == 0 else matrix.t()
    result = torch.linalg.solve_triangular(
        op_matrix,
        x_input.unsqueeze(1),
        upper=(uplo == 0 if trans == 0 else uplo == 1),
        unitriangular=(diag == 1),
    ).squeeze(1)
    x.copy_(result)
    return x


def parse_test_cases():
    test_cases = []
    for uplo, trans, diag, n, k, a_stride, x_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            default_a_stride = (1, k + 1) if k > 0 else (1, 2)

            a_spec = TensorSpec.from_tensor(
                (k + 1, n),
                a_stride if a_stride is not None else default_a_stride,
                dtype,
            )
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)

            test_cases.append(
                TestCase(
                    inputs=[a_spec, x_spec],
                    kwargs={"uplo": uplo, "trans": trans, "diag": diag, "k": k},
                    output_spec=None,
                    comparison_target=1,
                    tolerance=tol,
                    description="tbsv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 tbsv operator test"""

    def __init__(self):
        super().__init__("Tbsv")

    def get_test_cases(self):
        return parse_test_cases()

    # def torch_operator(self, *args, **kwargs):
    #     return torch_tbsv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.tbsv(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
