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

_UPLO_TRANS_CASES = [(uplo, trans) for uplo in (0,) for trans in (0,)]

_TEST_CASES_DATA = [
    # n, k, matrix_stride_n, matrix_stride_c, c_stride
    (128, 128, None, None, None),
    (1024, 64, None, None, None),
    (4096, 32, None, None, None),
    (5120, 16, None, None, None),
]

_TENSOR_DTYPES = [
    infinicore.complex64,
    # infinicore.complex128,
]

_REAL_DTYPE_MAP = {
    infinicore.complex64: infinicore.float32,
    infinicore.complex128: infinicore.float64,
}

_TOLERANCE_MAP = {
    infinicore.complex64: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.complex128: {"atol": 1e-9, "rtol": 1e-9},
}


def _default_col_major_stride(rows):
    return (1, rows)


def _matrix_stride(matrix_shape):
    if matrix_shape[0] == 1 and matrix_shape[1] > 1:
        return (1, 2)
    return _default_col_major_stride(matrix_shape[0])


def _triangle_update(c, update, uplo):
    update = update.clone()
    update.diagonal().imag.zero_()
    if uplo == 0:
        return torch.triu(update) + torch.tril(c, diagonal=-1)
    return torch.tril(update) + torch.triu(c, diagonal=1)


def _triangle_update_mlu(c, update_real, update_imag, uplo):
    idx = torch.arange(c.shape[0], device=c.device)
    update_imag[idx, idx] = 0

    if uplo == 0:
        out_real = torch.triu(update_real) + torch.tril(c.real, diagonal=-1)
        out_imag = torch.triu(update_imag) + torch.tril(c.imag, diagonal=-1)
    else:
        out_real = torch.tril(update_real) + torch.triu(c.real, diagonal=1)
        out_imag = torch.tril(update_imag) + torch.triu(c.imag, diagonal=1)

    out = torch.empty_like(c)
    out.real.copy_(out_real)
    out.imag.copy_(out_imag)
    return out


def _complex_mul(real_l, imag_l, real_r, imag_r):
    return real_l * real_r - imag_l * imag_r, real_l * imag_r + imag_l * real_r


def _her2k_update_mlu(alpha, a, b, beta, c, trans):
    a_real = a.real
    a_imag = a.imag
    b_real = b.real
    b_imag = b.imag
    alpha_real = alpha.real
    alpha_imag = alpha.imag

    if trans == 0:
        ab_real = torch.mm(a_real, b_real.t()) + torch.mm(a_imag, b_imag.t())
        ab_imag = torch.mm(a_imag, b_real.t()) - torch.mm(a_real, b_imag.t())
        ba_real = torch.mm(b_real, a_real.t()) + torch.mm(b_imag, a_imag.t())
        ba_imag = torch.mm(b_imag, a_real.t()) - torch.mm(b_real, a_imag.t())
    else:
        ab_real = torch.mm(a_real.t(), b_real) + torch.mm(a_imag.t(), b_imag)
        ab_imag = torch.mm(a_real.t(), b_imag) - torch.mm(a_imag.t(), b_real)
        ba_real = torch.mm(b_real.t(), a_real) + torch.mm(b_imag.t(), a_imag)
        ba_imag = torch.mm(b_real.t(), a_imag) - torch.mm(b_imag.t(), a_real)

    update_ab_real, update_ab_imag = _complex_mul(
        alpha_real, alpha_imag, ab_real, ab_imag
    )
    update_ba_real, update_ba_imag = _complex_mul(
        alpha_real, -alpha_imag, ba_real, ba_imag
    )
    update_real = update_ab_real + update_ba_real + beta * c.real
    update_imag = update_ab_imag + update_ba_imag + beta * c.imag
    return update_real, update_imag


def torch_her2k(a, b, alpha, beta, out, *, uplo=0, trans=0):
    if a.device.type == "mlu":
        update_real, update_imag = _her2k_update_mlu(alpha, a, b, beta, out, trans)
        out.copy_(_triangle_update_mlu(out, update_real, update_imag, uplo))
        return out

    if trans == 0:
        product = alpha * torch.mm(a, b.mH) + alpha.conj() * torch.mm(b, a.mH)
    else:
        product = alpha * torch.mm(a.mH, b) + alpha.conj() * torch.mm(b.mH, a)
    update = product + beta * out
    out.copy_(_triangle_update(out, update, uplo))
    return out


def parse_test_cases():
    test_cases = []
    for n, k, matrix_stride_n, matrix_stride_c, c_stride in _TEST_CASES_DATA:
        for uplo, trans in _UPLO_TRANS_CASES:
            for dtype in _TENSOR_DTYPES:
                matrix_shape = (n, k) if trans == 0 else (k, n)
                matrix_stride = matrix_stride_n if trans == 0 else matrix_stride_c
                if matrix_stride is None:
                    matrix_stride = _matrix_stride(matrix_shape)
                if c_stride is None:
                    c_stride = _default_col_major_stride(n)
                real_dtype = _REAL_DTYPE_MAP[dtype]
                tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

                a_spec = TensorSpec.from_tensor(
                    matrix_shape, matrix_stride, dtype, scale=0.5
                )
                b_spec = TensorSpec.from_tensor(
                    matrix_shape, matrix_stride, dtype, scale=0.5
                )
                alpha_spec = TensorSpec.from_tensor(
                    (), None, dtype, init_mode=TensorInitializer.ONES
                )
                beta_spec = TensorSpec.from_tensor(
                    (), None, real_dtype, init_mode=TensorInitializer.ONES
                )
                c_spec = TensorSpec.from_tensor(
                    (n, n), c_stride, dtype, init_mode=TensorInitializer.RANDOM
                )

                kwargs = {"uplo": uplo, "trans": trans}

                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec, alpha_spec, beta_spec, c_spec],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=4,
                        tolerance=tol,
                        description="her2k - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-3 her2k operator test"""

    def __init__(self):
        super().__init__("Her2k")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_her2k(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.her2k(*args, **kwargs)


def main():
    torch.manual_seed(0)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
