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
    # trans, m, n, kl, ku, a_stride, x_stride, y_stride
    (0, 128, 128, 2, 3, None, (2,), (3,)),
    (0, 1024, 1024, 4, 4, None, None, None),
    (0, 4096, 4096, 2, 2, None, (2,), None),
    (0, 5120, 5120, 3, 2, None, None, (2,)),
    (1, 128, 128, 2, 3, None, (2,), (3,)),
    (1, 1024, 1024, 4, 4, None, None, None),
    (1, 4096, 4096, 2, 2, None, None, (2,)),
    (1, 5120, 5120, 3, 2, None, (2,), None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _default_a_stride(kl, ku):
    return (1, kl + ku + 1)


def _full_from_band(a, m, n, kl, ku):
    full = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for j in range(n):
        i_begin = max(0, j - ku)
        i_end = min(m, j + kl + 1)
        for i in range(i_begin, i_end):
            full[i, j] = a[ku + i - j, j]
    return full


def torch_gbmv(alpha, a, x, beta, out, *, trans=0, kl, ku, m, n):
    full = _full_from_band(a, m, n, kl, ku)
    matrix = full if trans == 0 else full.t()
    result = alpha * torch.mv(matrix, x) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for trans, m, n, kl, ku, a_stride, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x_len = n if trans == 0 else m
            y_len = m if trans == 0 else n

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            a_spec = TensorSpec.from_tensor(
                (kl + ku + 1, n),
                a_stride if a_stride is not None else _default_a_stride(kl, ku),
                dtype,
            )
            x_spec = TensorSpec.from_tensor((x_len,), x_stride, dtype)
            beta_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            y_spec = TensorSpec.from_tensor(
                (y_len,), y_stride, dtype, init_mode=TensorInitializer.RANDOM
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, a_spec, x_spec, beta_spec, y_spec],
                    kwargs={"trans": trans, "kl": kl, "ku": ku, "m": m, "n": n},
                    output_spec=None,
                    comparison_target=4,
                    tolerance=tol,
                    description="gbmv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 gbmv operator test"""

    def __init__(self):
        super().__init__("Gbmv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_gbmv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("m")
        kwargs.pop("n")
        return infinicore.gbmv(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
