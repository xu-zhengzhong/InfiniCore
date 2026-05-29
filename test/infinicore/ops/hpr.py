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
    # uplo, n, x_stride
    (0, n, None)
    for n in (4096, 6144, 8192)
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
    raise ValueError(f"Unsupported Hpr dtype: {dtype}")


def _packed_indices(uplo, n, device):
    if uplo == 0:
        return torch.tril_indices(n, n, device=device)
    return torch.triu_indices(n, n, device=device)


def _packed_rank1_update(ap, x, alpha, uplo, n):
    rows, cols = _packed_indices(uplo, n, ap.device)
    update = alpha * torch.outer(x, x.conj())
    out = ap + update[cols, rows]

    diag = rows == cols
    out[diag] = out[diag].real.to(out.dtype)
    return out


def _packed_rank1_update_mlu(ap, x, alpha, uplo, n):
    rows, cols = _packed_indices(uplo, n, ap.device)
    x_real = x.real
    x_imag = x.imag
    update_real = alpha * (torch.outer(x_real, x_real) + torch.outer(x_imag, x_imag))
    update_imag = alpha * (torch.outer(x_imag, x_real) - torch.outer(x_real, x_imag))

    out = torch.empty_like(ap)
    out.real.copy_(ap.real + update_real[cols, rows])
    out.imag.copy_(ap.imag + update_imag[cols, rows])
    out.imag[rows == cols] = 0
    return out


def torch_hpr(alpha, x, out, *, uplo=0):
    n = x.shape[0]
    if out.device.type == "mlu":
        result = _packed_rank1_update_mlu(out, x, alpha, uplo, n)
    else:
        result = _packed_rank1_update(out, x, alpha, uplo, n)
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, x_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            packed_len = n * (n + 1) // 2

            alpha_spec = TensorSpec.from_tensor(
                (), None, _real_dtype(dtype), init_mode=TensorInitializer.ONES
            )
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            ap_spec = TensorSpec.from_tensor(
                (packed_len,), None, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, x_spec, ap_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=2,
                    tolerance=tol,
                    description="hpr - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 hpr operator test"""

    def __init__(self):
        super().__init__("Hpr")

    def get_test_cases(self):
        return parse_test_cases()

    # def torch_operator(self, *args, **kwargs):
    #     return torch_hpr(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.hpr(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
