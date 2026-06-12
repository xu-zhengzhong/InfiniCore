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
    # n, k, a_stride_n, a_stride_c, c_stride
    (128, 128, None, None, None),
    (1024, 64, None, None, None),
    (4096, 32, None, None, None),
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
    infinicore.complex128: {"atol": 1e-2, "rtol": 1e-2},
}


def _default_col_major_stride(rows):
    return (1, rows)


def _a_stride(a_shape):
    if a_shape[0] == 1 and a_shape[1] > 1:
        return (1, 2)
    return _default_col_major_stride(a_shape[0])


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


def _herk_update_mlu(alpha, a, beta, c, trans):
    a_real = a.real
    a_imag = a.imag

    if trans == 0:
        product_real = torch.mm(a_real, a_real.t()) + torch.mm(a_imag, a_imag.t())
        product_imag = torch.mm(a_imag, a_real.t()) - torch.mm(a_real, a_imag.t())
    else:
        product_real = torch.mm(a_real.t(), a_real) + torch.mm(a_imag.t(), a_imag)
        product_imag = torch.mm(a_real.t(), a_imag) - torch.mm(a_imag.t(), a_real)

    update_real = alpha * product_real + beta * c.real
    update_imag = alpha * product_imag + beta * c.imag
    return update_real, update_imag


def torch_herk(a, alpha, beta, out, *, uplo=0, trans=0):
    if a.device.type == "mlu":
        update_real, update_imag = _herk_update_mlu(alpha, a, beta, out, trans)
        out.copy_(_triangle_update_mlu(out, update_real, update_imag, uplo))
        return out

    product = torch.mm(a, a.mH) if trans == 0 else torch.mm(a.mH, a)
    update = alpha * product + beta * out
    out.copy_(_triangle_update(out, update, uplo))
    return out


def parse_test_cases():
    test_cases = []
    for n, k, a_stride_n, a_stride_c, c_stride in _TEST_CASES_DATA:
        for uplo, trans in _UPLO_TRANS_CASES:
            for dtype in _TENSOR_DTYPES:
                a_shape = (n, k) if trans == 0 else (k, n)
                a_stride = a_stride_n if trans == 0 else a_stride_c
                if a_stride is None:
                    a_stride = _a_stride(a_shape)
                if c_stride is None:
                    c_stride = _default_col_major_stride(n)
                real_dtype = _REAL_DTYPE_MAP[dtype]
                tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-2, "rtol": 1e-2})

                a_spec = TensorSpec.from_tensor(a_shape, a_stride, dtype, scale=0.5)
                alpha_spec = TensorSpec.from_tensor(
                    (), None, real_dtype, init_mode=TensorInitializer.ONES
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
                        inputs=[a_spec, alpha_spec, beta_spec, c_spec],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=3,
                        tolerance=tol,
                        description="herk - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-3 herk operator test"""

    def __init__(self):
        super().__init__("Herk")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_herk(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.herk(*args, **kwargs)


def main():
    torch.manual_seed(0)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
