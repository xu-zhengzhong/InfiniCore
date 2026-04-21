import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

# Test cases format: (shape, x_strides, y_strides, c, s)

_TEST_CASES_DATA = [
    ((13,), None, None, 0.8, 0.6),
    ((13,), (10,), (10,), 0.8, 0.6),
    ((5632,), None, None, 0.9238795, 0.38268343),
    ((5632,), (5,), (5,), 0.9238795, 0.38268343),
    ((16,), (4,), (4,), 0.9659258, 0.25881904),
    ((5632,), (32,), (32,), 0.9659258, 0.25881904),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.float64: {"atol": 1e-7, "rtol": 1e-6},
}

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]


def parse_test_cases():
    test_cases = []
    for shape, x_strides, y_strides, c, s in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)
            out_x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            out_y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[x_spec, y_spec],
                    kwargs={"c": c, "s": s},
                    output_spec=None,
                    output_count=2,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"Rot - OUT_OF_PLACE (c={c}, s={s})",
                )
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec, y_spec],
                    kwargs={"c": c, "s": s},
                    output_specs=[out_x_spec, out_y_spec],
                    comparison_target="out",
                    tolerance=tol,
                    description=f"Rot - INPLACE(out) (c={c}, s={s})",
                    output_count=2,
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Rot operator test (x, y -> c*x+s*y, c*y-s*x)"""

    def __init__(self):
        super().__init__("Rot")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        x, y = args
        c = kwargs.pop("c")
        s = kwargs.pop("s")
        out = kwargs.pop("out", None)
        out_x = x.clone()
        out_y = y.clone()
        x0 = out_x.clone()
        y0 = out_y.clone()
        out_x.copy_(c * x0 + s * y0)
        out_y.copy_(c * y0 - s * x0)
        if out is not None:
            out[0].copy_(out_x)
            out[1].copy_(out_y)
            return out
        return out_x, out_y

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.rot(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()