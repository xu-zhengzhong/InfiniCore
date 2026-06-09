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
    (0, 1024, 5, None, None, None),
    (0, 4096, 3, None, (2,), None),
    (0, 5120, 2, None, None, (2,)),
    (1, 128, 7, None, (3,), (2,)),
    (1, 1024, 6, None, None, None),
    (1, 4096, 4, None, None, (2,)),
    (1, 5120, 3, None, (2,), None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _full_from_symmetric_band(a, n, k, uplo):
    full = torch.zeros((n, n), dtype=a.dtype, device=a.device)
    if uplo == 0:
        for j in range(n):
            i_begin = max(0, j - k)
            for i in range(i_begin, j + 1):
                value = a[k + i - j, j]
                full[i, j] = value
                full[j, i] = value
    else:
        for j in range(n):
            i_end = min(n, j + k + 1)
            for i in range(j, i_end):
                value = a[i - j, j]
                full[i, j] = value
                full[j, i] = value
    return full


def torch_sbmv(alpha, a, x, beta, out, *, uplo=0, k=0):
    matrix = _full_from_symmetric_band(a, x.shape[0], k, uplo)
    result = alpha * torch.mv(matrix, x) + beta * out
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, k, a_stride, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
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
                    description="sbmv - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 sbmv operator test"""

    def __init__(self):
        super().__init__("Sbmv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_sbmv(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.sbmv(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
