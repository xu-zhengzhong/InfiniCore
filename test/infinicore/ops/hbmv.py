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
    # uplo, n, k, a_stride, x_stride, y_stride
    (0, 128, 7, None, (2,), (3,)),
    (0, 1024, 6, None, None, None),
    (0, 4096, 4, None, (2,), None),
    (1, 128, 7, None, (3,), (2,)),
    (1, 1024, 8, None, None, None),
    (1, 4096, 5, None, None, (2,)),
]

_TENSOR_DTYPES = [
    infinicore.complex64,
    # infinicore.complex128,
]

_TOLERANCE_MAP = {
    infinicore.complex64: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.complex128: {"atol": 1e-3, "rtol": 1e-3},
}


def _full_from_hermitian_band(a, n, k, uplo):
    full = torch.zeros((n, n), dtype=a.dtype, device=a.device)
    if uplo == 0:
        for j in range(n):
            i_begin = max(0, j - k)
            for i in range(i_begin, j + 1):
                value = a[k + i - j, j]
                if i == j:
                    value = value.real.to(a.dtype)
                full[i, j] = value
                full[j, i] = value.conj()
    else:
        for j in range(n):
            i_end = min(n, j + k + 1)
            for i in range(j, i_end):
                value = a[i - j, j]
                if i == j:
                    value = value.real.to(a.dtype)
                full[i, j] = value
                full[j, i] = value.conj()
    return full


def _full_from_hermitian_band_mlu(a, n, k, uplo):
    full_real = torch.zeros((n, n), dtype=a.real.dtype, device=a.device)
    full_imag = torch.zeros((n, n), dtype=a.real.dtype, device=a.device)
    a_real = a.real
    a_imag = a.imag

    if uplo == 0:
        for j in range(n):
            i_begin = max(0, j - k)
            for i in range(i_begin, j + 1):
                value_real = a_real[k + i - j, j]
                value_imag = a_imag[k + i - j, j]
                if i == j:
                    value_imag = value_imag * 0
                full_real[i, j] = value_real
                full_real[j, i] = value_real
                full_imag[i, j] = value_imag
                full_imag[j, i] = -value_imag
    else:
        for j in range(n):
            i_end = min(n, j + k + 1)
            for i in range(j, i_end):
                value_real = a_real[i - j, j]
                value_imag = a_imag[i - j, j]
                if i == j:
                    value_imag = value_imag * 0
                full_real[i, j] = value_real
                full_real[j, i] = value_real
                full_imag[i, j] = value_imag
                full_imag[j, i] = -value_imag

    return full_real, full_imag


def _hbmv_mlu(alpha, a, x, beta, out, uplo, k):
    matrix_real, matrix_imag = _full_from_hermitian_band_mlu(a, x.shape[0], k, uplo)
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


def torch_hbmv(alpha, a, x, beta, out, *, uplo=0, k=0):
    if a.device.type == "mlu":
        result = _hbmv_mlu(alpha, a, x, beta, out, uplo, k)
        out.copy_(result)
        return out

    matrix = _full_from_hermitian_band(a, x.shape[0], k, uplo)
    result = alpha * torch.mv(matrix, x) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, k, a_stride, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})
            default_a_stride = (1, k + 1) if k > 0 else (1, 2)

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            a_spec = TensorSpec.from_tensor(
                (k + 1, n),
                a_stride if a_stride is not None else default_a_stride,
                dtype,
            )
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            beta_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            y_spec = TensorSpec.from_tensor(
                (n,), y_stride, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, a_spec, x_spec, beta_spec, y_spec],
                    kwargs={"uplo": uplo, "k": k},
                    output_spec=None,
                    comparison_target=4,
                    tolerance=tol,
                    description="hbmv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 hbmv operator test"""

    def __init__(self):
        super().__init__("Hbmv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_hbmv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.hbmv(*args, **kwargs)


def main():
    torch.manual_seed(0)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
