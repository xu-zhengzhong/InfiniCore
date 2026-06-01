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
    # uplo, n, x_stride, y_stride
    (0, 1, None, None),
    (0, 5, None, None),
    (0, 17, (2,), None),
    (0, 33, None, (2,)),
    (0, 128, (2,), (3,)),
    (0, 256, None, None),
    (1, 1, None, None),
    (1, 5, None, None),
    (1, 17, None, (2,)),
    (1, 33, (2,), None),
    (1, 128, (3,), (2,)),
    (1, 256, None, None),
]

_TENSOR_DTYPES = [
    infinicore.complex64,
    # infinicore.complex128,
]

_TOLERANCE_MAP = {
    infinicore.complex64: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.complex128: {"atol": 1e-9, "rtol": 1e-9},
}


def _packed_indices(uplo, n, device):
    if uplo == 0:
        return torch.tril_indices(n, n, device=device)
    return torch.triu_indices(n, n, device=device)


def _packed_rank2_update(ap, x, y, alpha, uplo, n):
    rows, cols = _packed_indices(uplo, n, ap.device)
    update = alpha * torch.outer(x, y.conj()) + alpha.conj() * torch.outer(y, x.conj())
    out = ap + update[cols, rows]

    diag = rows == cols
    out[diag] = out[diag].real.to(out.dtype)
    return out


def _packed_rank2_update_mlu(ap, x, y, alpha, uplo, n):
    rows, cols = _packed_indices(uplo, n, ap.device)
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

    out = torch.empty_like(ap)
    out.real.copy_(ap.real + update_real[cols, rows])
    out.imag.copy_(ap.imag + update_imag[cols, rows])
    out.imag[rows == cols] = 0
    return out


def torch_hpr2(alpha, x, y, out, *, uplo=0):
    n = x.shape[0]
    if out.device.type == "mlu":
        result = _packed_rank2_update_mlu(out, x, y, alpha, uplo, n)
    else:
        result = _packed_rank2_update(out, x, y, alpha, uplo, n)
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            packed_len = n * (n + 1) // 2

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.RANDOM
            )
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            y_spec = TensorSpec.from_tensor((n,), y_stride, dtype)
            ap_spec = TensorSpec.from_tensor(
                (packed_len,), None, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, x_spec, y_spec, ap_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=3,
                    tolerance=tol,
                    description="hpr2 - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 hpr2 operator test"""

    def __init__(self):
        super().__init__("Hpr2")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_hpr2(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.hpr2(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
