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
    # uplo, n, a_stride, x_stride
    (0, 128, None, (3,)),
    (0, 1024, None, None),
    (0, 4096, None, (2,)),
    (0, 5120, None, None),
    (1, 128, None, (3,)),
    (1, 1024, None, None),
    (1, 4096, None, None),
    (1, 5120, None, (2,)),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _triangle_update(a, update, uplo):
    if uplo == 0:
        return torch.triu(a + update) + torch.tril(a, diagonal=-1)
    return torch.tril(a + update) + torch.triu(a, diagonal=1)


def torch_syr(alpha, x, out, *, uplo=0):
    result = _triangle_update(out, alpha * torch.outer(x, x), uplo)
    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for uplo, n, a_stride, x_stride in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            x_spec = TensorSpec.from_tensor((n,), x_stride, dtype)
            a_spec = TensorSpec.from_tensor(
                (n, n),
                a_stride if a_stride is not None else (1, n),
                dtype,
                init_mode=TensorInitializer.RANDOM,
            )

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, x_spec, a_spec],
                    kwargs={"uplo": uplo},
                    output_spec=None,
                    comparison_target=2,
                    tolerance=tol,
                    description="syr - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-2 syr operator test"""

    def __init__(self):
        super().__init__("Syr")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_syr(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.syr(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
