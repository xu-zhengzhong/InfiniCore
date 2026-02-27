import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)
from framework.tensor import TensorInitializer

# Test cases format: (in_shape, in_strides_or_None, q_shape_or_value, dim_or_None, keepdim_or_None, interpolation, out_strides_or_None)
# quantile computes quantiles along dim or overall. q may be float or tensor

_TEST_CASES_DATA = [
    ((8, 8), None, 0.5, None, None, "linear", None),
    ((8, 8), None, (2,), None, None, "linear", None),
    ((8, 8), (16, 1), 0.25, 1, False, "lower", None),
    ((8, 8), (16, 1), (3,), 1, False, "lower", None),
    ((2, 3, 4), None, 0.75, 2, True, "higher", (0, 1, 1)),
    ((2, 3, 4), None, (4,), 2, True, "higher", (6, 3, 1, 1)),
    ((16, 64), (128, 1), 0.5, None, None, "nearest", None),
    ((16, 64), (128, 1), (5,), None, None, "nearest", None),
    ((4, 5, 6), (60, 12, 2), 0.5, 2, True, "midpoint", (12, 4, 1)),
    ((4, 5, 6), (60, 12, 2), (1,), 2, True, "midpoint", (200, 12, 4, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float32]


def _compute_out_shape(shape, dim, keepdim, q):
    # if q is tensor with len>1, output shape may include q dim; keep simple: when q is tensor, return (len(q), ...) prefix
    if dim is None:
        base = ()
    else:
        if isinstance(dim, tuple):
            dims = sorted([(d if d >= 0 else len(shape) + d) for d in dim])
        else:
            dims = [dim]
        if keepdim:
            out = list(shape)
            for d in dims:
                out[d] = 1
            base = tuple(out)
        else:
            base = tuple(s for i, s in enumerate(shape) if i not in dims)

    if isinstance(q, tuple):
        # Prepend q-length as first dim
        return (q[0],) + base
    return base


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, q_shape_or_value, dim, keepdim, interpolation, out_strides = data
        # q_is_tensor = isinstance(q, torch.Tensor)
        out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {}
            if isinstance(q_shape_or_value, tuple):
                q_spec = TensorSpec.from_tensor(q_shape_or_value, dtype=dtype, init_mode=TensorInitializer.RANDOM)
                inputs = [in_spec, q_spec]
            else:
                kwargs = {"q": q_shape_or_value}
                inputs = [in_spec]

            if interpolation is not None:
                kwargs["interpolation"] = interpolation
            if dim is not None:
                kwargs["dim"] = dim
            if keepdim is not None:
                kwargs["keepdim"] = keepdim

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Quantile - OUT_OF_PLACE",
                )
            )

            out_shape = _compute_out_shape(shape, dim, keepdim, q_shape_or_value)
            out_spec = TensorSpec.from_tensor(out_shape, out_strides, dtype)
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=inputs,
                        kwargs=kwargs,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="Quantile - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Quantile operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Quantile")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.quantile(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.quantile(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
