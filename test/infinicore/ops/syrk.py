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

_UPLO_TRANS_CASES = [(uplo, trans) for uplo in (0,) for trans in (1,)]

_TEST_CASES_DATA = [
    # n, k, a_stride_n, a_stride_t, c_stride
    (128, 128, None, None, None),
    (1024, 1024, None, None, None),
    (4096, 4096, None, None, None),
    (5120, 5120, None, None, None),
]

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 5e-4, "rtol": 5e-4},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
}


def _default_col_major_stride(rows):
    return (1, rows)


def _triangle_update(c, update, uplo):
    if uplo == 0:
        return torch.triu(update) + torch.tril(c, diagonal=-1)
    return torch.tril(update) + torch.triu(c, diagonal=1)


def torch_syrk(a, alpha, beta, out, *, uplo=0, trans=0):
    product = torch.mm(a, a.t()) if trans == 0 else torch.mm(a.t(), a)
    update = alpha * product + beta * out
    out.copy_(_triangle_update(out, update, uplo))
    return out


def parse_test_cases():
    test_cases = []
    for n, k, a_stride_n, a_stride_t, c_stride in _TEST_CASES_DATA:
        for uplo, trans in _UPLO_TRANS_CASES:
            for dtype in _TENSOR_DTYPES:
                a_shape = (n, k) if trans == 0 else (k, n)
                a_stride = a_stride_n if trans == 0 else a_stride_t
                if a_stride is None:
                    a_stride = _default_col_major_stride(a_shape[0])
                if c_stride is None:
                    c_stride = _default_col_major_stride(n)
                tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

                a_spec = TensorSpec.from_tensor(a_shape, a_stride, dtype)
                alpha_spec = TensorSpec.from_tensor(
                    (), None, dtype, init_mode=TensorInitializer.ONES
                )
                beta_spec = TensorSpec.from_tensor(
                    (), None, dtype, init_mode=TensorInitializer.ONES
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
                        description="syrk - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-3 syrk operator test"""

    def __init__(self):
        super().__init__("Syrk")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_syrk(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.syrk(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
