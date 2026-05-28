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
    (2, 2, None, None, None, None),
    (3, 5, None, None, None, None),
    (5, 3, None, None, None, None),
    (8, 8, None, None, None, None),
    (9, 17, None, None, None, None),
    (17, 9, None, None, None, None),
    (31, 32, None, None, None, None),
    (32, 31, None, None, None, None),
    (65, 65, None, None, None, None),
    (127, 128, None, None, None, None),
    (128, 127, None, None, None, None),
    (256, 256, None, None, None, None),
    (512, 512, None, None, None, None),
    (1024, 1024, None, None, None, None),
    (2, 2048, None, None, None, None),
    (4096, 2, None, None, None, None),
    (17, 9, (1, 24), (1, 16), None, None),
    (31, 32, (1, 40), (1, 48), (1, 36), (1, 40)),
    (4096, 2, (1, 4104), (1, 4), (1, 4100), (1, 4104)),
    (2, 2048, (1, 4), (1, 2050), (1, 4), (1, 4)),
]

_TENSOR_DTYPES = [
    infinicore.complex64,
    # infinicore.complex128,
]

_TOLERANCE_MAP = {
    infinicore.complex64: {"atol": 3e-3, "rtol": 5e-4},
}


def _full_from_triangle(a, uplo):
    if uplo == 0:
        tri = torch.triu(a)
        matrix = tri + torch.triu(a, diagonal=1).mH
    else:
        tri = torch.tril(a)
        matrix = tri + torch.tril(a, diagonal=-1).mH
    diag = torch.diagonal(matrix)
    diag.copy_(diag.real.clone())
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


def _hemm_mlu(a, b, alpha, beta, out, side, uplo):
    a_real, a_imag = _full_from_triangle_mlu(a, uplo)
    b_real = b.real
    b_imag = b.imag

    if side == 0:
        product_real = torch.mm(a_real, b_real) - torch.mm(a_imag, b_imag)
        product_imag = torch.mm(a_real, b_imag) + torch.mm(a_imag, b_real)
    else:
        product_real = torch.mm(b_real, a_real) - torch.mm(b_imag, a_imag)
        product_imag = torch.mm(b_real, a_imag) + torch.mm(b_imag, a_real)

    alpha_real = alpha.real
    alpha_imag = alpha.imag
    beta_real = beta.real
    beta_imag = beta.imag
    out_real = out.real
    out_imag = out.imag

    result_real = alpha_real * product_real - alpha_imag * product_imag
    result_real = result_real + beta_real * out_real - beta_imag * out_imag
    result_imag = alpha_real * product_imag + alpha_imag * product_real
    result_imag = result_imag + beta_real * out_imag + beta_imag * out_real

    result = torch.empty_like(out)
    result.real.copy_(result_real)
    result.imag.copy_(result_imag)
    return result


def torch_hemm(a, b, alpha, beta, out, *, side=0, uplo=0):
    if a.device.type == "mlu":
        result = _hemm_mlu(a, b, alpha, beta, out, side, uplo)
        out.copy_(result)
        return out

    matrix = _full_from_triangle(a, uplo)
    product = torch.mm(matrix, b) if side == 0 else torch.mm(b, matrix)
    result = alpha * product + beta * out
    out.copy_(result)
    return out


def _default_col_major_stride(rows):
    return (1, rows)


def parse_test_cases():
    test_cases = []
    for m, n, a_stride_left, a_stride_right, b_stride, c_stride in _TEST_CASES_DATA:
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
                    (), None, dtype, init_mode=TensorInitializer.ONES
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
                        description="hemm - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-3 hemm operator test"""

    def __init__(self):
        super().__init__("Hemm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_hemm(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.hemm(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
