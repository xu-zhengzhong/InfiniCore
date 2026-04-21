import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

_TEST_CASES_DATA = [
    (0.0, 0.0),
    (3.0, 4.0),
    (-2.5, 5.0),
    (7.0, -1.5),
    (-3.2, -8.4),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-6, "rtol": 1e-6},
    infinicore.float64: {"atol": 1e-12, "rtol": 1e-12},
}

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]


def parse_test_cases():
    test_cases = []
    for a0, b0 in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            a_spec = TensorSpec.from_tensor((1,), None, dtype)
            b_spec = TensorSpec.from_tensor((1,), None, dtype)
            out_a_spec = TensorSpec.from_tensor((1,), None, dtype)
            out_b_spec = TensorSpec.from_tensor((1,), None, dtype)
            out_c_spec = TensorSpec.from_tensor((1,), None, dtype)
            out_s_spec = TensorSpec.from_tensor((1,), None, dtype)

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_count=4,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"Rotg - OUT_OF_PLACE (a={a0}, b={b0})",
                )
            )

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_specs=[out_a_spec, out_b_spec, out_c_spec, out_s_spec],
                    comparison_target="out",
                    tolerance=tol,
                    output_count=4,
                    description=f"Rotg - INPLACE(out) (a={a0}, b={b0})",
                )
            )

    return test_cases


def torch_rotg(a, b):
    a0 = float(a.item())
    b0 = float(b.item())
    anorm = abs(a0)
    bnorm = abs(b0)

    if bnorm == 0.0:
        return (
            torch.tensor([a0], dtype=a.dtype, device=a.device),
            torch.tensor([0.0], dtype=a.dtype, device=a.device),
            torch.tensor([1.0], dtype=a.dtype, device=a.device),
            torch.tensor([0.0], dtype=a.dtype, device=a.device),
        )
    if anorm == 0.0:
        return (
            torch.tensor([b0], dtype=a.dtype, device=a.device),
            torch.tensor([1.0], dtype=a.dtype, device=a.device),
            torch.tensor([0.0], dtype=a.dtype, device=a.device),
            torch.tensor([1.0], dtype=a.dtype, device=a.device),
        )

    sigma = math.copysign(1.0, a0 if anorm > bnorm else b0)
    r = sigma * math.hypot(a0, b0)
    c = a0 / r
    s = b0 / r
    if anorm > bnorm:
        z = s
    elif c != 0.0:
        z = 1.0 / c
    else:
        z = 1.0

    return (
        torch.tensor([r], dtype=a.dtype, device=a.device),
        torch.tensor([z], dtype=a.dtype, device=a.device),
        torch.tensor([c], dtype=a.dtype, device=a.device),
        torch.tensor([s], dtype=a.dtype, device=a.device),
    )


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Rotg")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, a, b, **kwargs):
        out = kwargs.pop("out", None)
        out_a, out_b, out_c, out_s = torch_rotg(a, b)
        if out is not None:
            out[0].copy_(out_a)
            out[1].copy_(out_b)
            out[2].copy_(out_c)
            out[3].copy_(out_s)
            return out
        return out_a, out_b, out_c, out_s

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.rotg(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()