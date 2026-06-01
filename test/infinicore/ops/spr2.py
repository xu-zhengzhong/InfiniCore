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
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _packed_rank2_update(ap, x, y, alpha, uplo, n):
    update = alpha * (torch.outer(x, y) + torch.outer(y, x))
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=ap.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=ap.device)

    return ap + update[cols, rows]


def torch_spr2(alpha, x, y, out, *, uplo=0):
    result = _packed_rank2_update(out, x, y, alpha, uplo, x.shape[0])
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, x_stride, y_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            packed_len = n * (n + 1) // 2

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
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
                    description="spr2 - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 spr2 operator test"""

    def __init__(self):
        super().__init__("Spr2")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_spr2(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.spr2(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
