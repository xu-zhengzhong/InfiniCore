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
    # uplo, n, a_stride, x_stride, y_stride
    (0, 128, None, (2,), (3,)),
    (0, 1024, None, None, None),
    (0, 4096, None, (2,), None),
    (0, 5120, None, None, (2,)),
    (1, 128, None, (3,), (2,)),
    (1, 1024, None, None, None),
    (1, 4096, None, None, (2,)),
    (1, 5120, None, (2,), None),
]

_TENSOR_DTYPES = [
    infinicore.complex64,
    # infinicore.complex128,
]

_TOLERANCE_MAP = {
    infinicore.complex64: {"atol": 5e-4, "rtol": 5e-4},
    infinicore.complex128: {"atol": 1e-9, "rtol": 1e-9},
}


def _full_from_triangle(a, uplo):
    if uplo == 0:
        matrix = torch.triu(a) + torch.triu(a, diagonal=1).mH
    else:
        matrix = torch.tril(a) + torch.tril(a, diagonal=-1).mH

    idx = torch.arange(a.shape[0], device=a.device)
    matrix[idx, idx] = matrix[idx, idx].real.to(matrix.dtype)
    return matrix


def _full_from_triangle_mlu(a, uplo):
    a_real = a.real
    a_imag = a.imag

    if uplo == 0:
        matrix_real = torch.triu(a_real) + torch.triu(a_real, diagonal=1).t()
        matrix_imag = torch.triu(a_imag) - torch.triu(a_imag, diagonal=1).t()
    else:
        matrix_real = torch.tril(a_real) + torch.tril(a_real, diagonal=-1).t()
        matrix_imag = torch.tril(a_imag) - torch.tril(a_imag, diagonal=-1).t()

    idx = torch.arange(a.shape[0], device=a.device)
    matrix_imag[idx, idx] = 0
    return matrix_real, matrix_imag


def _hemv_mlu(alpha, a, x, beta, out, uplo):
    matrix_real, matrix_imag = _full_from_triangle_mlu(a, uplo)
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


def torch_hemv(alpha, a, x, beta, out, *, uplo=0):
    if a.device.type == "mlu":
        result = _hemv_mlu(alpha, a, x, beta, out, uplo)
        out.copy_(result)
        return out

    matrix = _full_from_triangle(a, uplo)
    result = alpha * torch.mv(matrix, x) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, a_stride, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 5e-4, "rtol": 5e-4})

            alpha_spec = TensorSpec.from_tensor((), None, dtype)
            a_spec = TensorSpec.from_tensor(
                (n, n), a_stride if a_stride is not None else (1, n), dtype
            )
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            beta_spec = TensorSpec.from_tensor((), None, dtype)
            y_spec = TensorSpec.from_tensor((n,), y_stride, dtype)

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, a_spec, x_spec, beta_spec, y_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=4,
                    tolerance=tol,
                    description="hemv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 hemv operator test"""

    def __init__(self):
        super().__init__("Hemv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_hemv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.hemv(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
