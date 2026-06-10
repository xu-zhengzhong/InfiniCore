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
    # uplo, n, x_stride, y_stride
    (0, 128, (2,), (3,)),
    (0, 1024, None, None),
    (0, 4096, (2,), None),
    (0, 5120, None, (2,)),
    (1, 128, (3,), (2,)),
    (1, 1024, None, None),
    (1, 4096, None, (2,)),
    (1, 5120, (2,), None),
]

_TENSOR_DTYPES = [
    infinicore.complex64,
    # infinicore.complex128,
]

_TOLERANCE_MAP = {
    infinicore.complex64: {"atol": 5e-4, "rtol": 5e-4},
    infinicore.complex128: {"atol": 1e-9, "rtol": 1e-9},
}


def _packed_to_full(ap, uplo, n):
    matrix = torch.empty((n, n), dtype=ap.dtype, device=ap.device)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=ap.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=ap.device)
    matrix[cols, rows] = ap
    matrix[rows, cols] = ap.conj()

    idx = torch.arange(n, device=ap.device)
    matrix[idx, idx] = matrix[idx, idx].real.to(matrix.dtype)
    return matrix


def _packed_to_full_mlu(ap, uplo, n):
    matrix_real = torch.empty((n, n), dtype=ap.real.dtype, device=ap.device)
    matrix_imag = torch.empty((n, n), dtype=ap.real.dtype, device=ap.device)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=ap.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=ap.device)

    matrix_real[cols, rows] = ap.real
    matrix_real[rows, cols] = ap.real
    matrix_imag[cols, rows] = ap.imag
    matrix_imag[rows, cols] = -ap.imag

    idx = torch.arange(n, device=ap.device)
    matrix_imag[idx, idx] = 0
    return matrix_real, matrix_imag


def _hpmv_mlu(alpha, ap, x, beta, out, uplo, n):
    matrix_real, matrix_imag = _packed_to_full_mlu(ap, uplo, n)
    x_real = x.real
    x_imag = x.imag

    mv_real = torch.mv(matrix_real, x_real) - torch.mv(matrix_imag, x_imag)
    mv_imag = torch.mv(matrix_real, x_imag) + torch.mv(matrix_imag, x_real)

    alpha_real = alpha.real
    alpha_imag = alpha.imag
    beta_real = beta.real
    beta_imag = beta.imag
    out_real = out.real
    out_imag = out.imag

    result_real = alpha_real * mv_real - alpha_imag * mv_imag
    result_real = result_real + beta_real * out_real - beta_imag * out_imag
    result_imag = alpha_real * mv_imag + alpha_imag * mv_real
    result_imag = result_imag + beta_real * out_imag + beta_imag * out_real

    result = torch.empty_like(out)
    result.real.copy_(result_real)
    result.imag.copy_(result_imag)
    return result


def torch_hpmv(alpha, ap, x, beta, out, *, uplo=0):
    n = x.shape[0]
    if ap.device.type == "mlu":
        result = _hpmv_mlu(alpha, ap, x, beta, out, uplo, n)
        out.copy_(result)
        return out

    matrix = _packed_to_full(ap, uplo, n)
    result = alpha * torch.mv(matrix, x) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 5e-4, "rtol": 5e-4})
            packed_len = n * (n + 1) // 2

            alpha_spec = TensorSpec.from_tensor((), None, dtype)
            ap_spec = TensorSpec.from_tensor((packed_len,), None, dtype)
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            beta_spec = TensorSpec.from_tensor((), None, dtype)
            y_spec = TensorSpec.from_tensor((n,), y_stride, dtype)

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, ap_spec, x_spec, beta_spec, y_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=4,
                    tolerance=tol,
                    description="hpmv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 hpmv operator test"""

    def __init__(self):
        super().__init__("Hpmv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_hpmv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.hpmv(*args, **kwargs)


def main():
    torch.manual_seed(0)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
