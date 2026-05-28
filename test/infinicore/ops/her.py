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
    # uplo, n, a_stride, x_stride
    (0, 1, None, None),
    (0, 5, None, None),
    (0, 17, None, (2,)),
    (0, 33, (1, 40), None),
    (0, 128, None, (3,)),
    (0, 1024, None, None),
    (1, 1, None, None),
    (1, 5, None, None),
    (1, 17, None, (2,)),
    (1, 33, (1, 40), (2,)),
    (1, 128, None, (3,)),
    (1, 1024, None, None),
]

_TENSOR_DTYPES = [
    infinicore.complex64,
    # infinicore.complex128,
]

_TOLERANCE_MAP = {
    infinicore.complex64: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.complex128: {"atol": 1e-9, "rtol": 1e-9},
}


def _real_dtype(dtype):
    if dtype == infinicore.complex64:
        return infinicore.float32
    if dtype == infinicore.complex128:
        return infinicore.float64
    raise ValueError(f"Unsupported Her dtype: {dtype}")


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


def torch_her(alpha, x, out, *, uplo=0):
    if out.device.type == "mlu":
        x_real = x.real
        x_imag = x.imag
        update_real = alpha * (
            torch.outer(x_real, x_real) + torch.outer(x_imag, x_imag)
        )
        update_imag = alpha * (
            torch.outer(x_imag, x_real) - torch.outer(x_real, x_imag)
        )
        result = _triangle_update_mlu(out, update_real, update_imag, uplo)
    else:
        result = _triangle_update(out, alpha * torch.outer(x, x.conj()), uplo)

    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, a_stride, x_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-5})

            alpha_spec = TensorSpec.from_tensor((), None, _real_dtype(dtype))
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            a_spec = TensorSpec.from_tensor(
                (n, n), a_stride if a_stride is not None else (1, n), dtype
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, x_spec, a_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=2,
                    tolerance=tol,
                    description="her - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 her operator test"""

    def __init__(self):
        super().__init__("Her")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_her(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.her(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
