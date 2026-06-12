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
    (1, 128, None, (3,), (2,)),
    (1, 1024, None, None, None),
    (1, 4096, None, None, (2,)),
]

_TENSOR_DTYPES = [
    infinicore.complex64,
    # infinicore.complex128,
]

_TOLERANCE_MAP = {
    infinicore.complex64: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.complex128: {"atol": 1e-3, "rtol": 1e-3},
}


def _triangle_update(a, update, uplo):
    if uplo == 0:
        out = torch.triu(a + update) + torch.tril(a, diagonal=-1)
    else:
        out = torch.tril(a + update) + torch.triu(a, diagonal=1)

    idx = torch.arange(a.shape[0], device=a.device)
    out[idx, idx] = out[idx, idx].real.to(out.dtype)
    return out


def _triangle_update_mlu(a, update_real, update_imag, uplo):
    a_real = a.real
    a_imag = a.imag

    if uplo == 0:
        out_real = torch.triu(a_real + update_real) + torch.tril(a_real, diagonal=-1)
        out_imag = torch.triu(a_imag + update_imag) + torch.tril(a_imag, diagonal=-1)
    else:
        out_real = torch.tril(a_real + update_real) + torch.triu(a_real, diagonal=1)
        out_imag = torch.tril(a_imag + update_imag) + torch.triu(a_imag, diagonal=1)

    idx = torch.arange(a.shape[0], device=a.device)
    out_imag[idx, idx] = 0

    out = torch.empty_like(a)
    out.real.copy_(out_real)
    out.imag.copy_(out_imag)
    return out


def torch_her2(alpha, x, y, out, *, uplo=0):
    if out.device.type == "mlu":
        x_real = x.real
        x_imag = x.imag
        y_real = y.real
        y_imag = y.imag
        alpha_real = alpha.real
        alpha_imag = alpha.imag

        xyh_real = torch.outer(x_real, y_real) + torch.outer(x_imag, y_imag)
        xyh_imag = torch.outer(x_imag, y_real) - torch.outer(x_real, y_imag)
        yxh_real = torch.outer(y_real, x_real) + torch.outer(y_imag, x_imag)
        yxh_imag = torch.outer(y_imag, x_real) - torch.outer(y_real, x_imag)

        update_real = alpha_real * xyh_real - alpha_imag * xyh_imag
        update_real = update_real + alpha_real * yxh_real + alpha_imag * yxh_imag
        update_imag = alpha_real * xyh_imag + alpha_imag * xyh_real
        update_imag = update_imag + alpha_real * yxh_imag - alpha_imag * yxh_real
        result = _triangle_update_mlu(out, update_real, update_imag, uplo)
    else:
        update = alpha * torch.outer(x, y.conj()) + alpha.conj() * torch.outer(
            y, x.conj()
        )
        result = _triangle_update(out, update, uplo)

    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, a_stride, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})

            alpha_spec = TensorSpec.from_tensor((), None, dtype)
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            y_spec = TensorSpec.from_tensor((n,), y_stride, dtype)
            a_spec = TensorSpec.from_tensor(
                (n, n), a_stride if a_stride is not None else (1, n), dtype
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, x_spec, y_spec, a_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=3,
                    tolerance=tol,
                    description="her2 - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 her2 operator test"""

    def __init__(self):
        super().__init__("Her2")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_her2(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.her2(*args, **kwargs)


def main():
    torch.manual_seed(0)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
