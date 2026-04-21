import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
    convert_infinicore_to_torch,
    infinicore_tensor_from_torch,
)

_TEST_CASES_DATA = [
    (1.0, 2.0, 3.0, 4.0),
    (2.5, 0.5, -1.2, 0.8),
    (3.0, 4.0, 0.0, 2.0),
    (1.5, 1.5, 2.0, -3.0),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-12, "rtol": 1e-12},
}

_TENSOR_DTYPES = [
    infinicore.float32,
    # infinicore.float64,
]


def torch_rotmg(d1, d2, x1, y1):
    zero = 0.0
    one = 1.0
    two = 2.0
    gam = 4096.0
    gamsq = 1.67772e7
    rgamsq = 5.96046e-8

    sparam = [0.0] * 5
    sh11 = sh12 = sh21 = sh22 = 0.0

    if d1 < zero:
        sflag = -one
        d1 = d2 = x1 = zero
    else:
        sp2 = d2 * y1
        if sp2 == zero:
            sparam[0] = -two
            return d1, d2, x1, sparam

        sp1 = d1 * x1
        sq2 = sp2 * y1
        sq1 = sp1 * x1

        if abs(sq1) > abs(sq2):
            sh21 = -y1 / x1
            sh12 = sp2 / sp1
            su = one - sh12 * sh21
            if su > zero:
                sflag = zero
                d1 = d1 / su
                d2 = d2 / su
                x1 = x1 * su
            else:
                sflag = -one
                sh11 = sh12 = sh21 = sh22 = zero
                d1 = d2 = x1 = zero
        else:
            if sq2 < zero:
                sflag = -one
                d1 = d2 = x1 = zero
            else:
                sflag = one
                sh11 = sp1 / sp2
                sh22 = x1 / y1
                su = one + sh11 * sh22
                stemp = d2 / su
                d2 = d1 / su
                d1 = stemp
                x1 = y1 * su

        if d1 != zero:
            while d1 <= rgamsq or d1 >= gamsq:
                if sflag == zero:
                    sh11 = one
                    sh22 = one
                    sflag = -one
                else:
                    sh21 = -one
                    sh12 = one
                    sflag = -one
                if d1 <= rgamsq:
                    d1 = d1 * gam * gam
                    x1 = x1 / gam
                    sh11 = sh11 / gam
                    sh12 = sh12 / gam
                else:
                    d1 = d1 / (gam * gam)
                    x1 = x1 * gam
                    sh11 = sh11 * gam
                    sh12 = sh12 * gam

        if d2 != zero:
            while abs(d2) <= rgamsq or abs(d2) >= gamsq:
                if sflag == zero:
                    sh11 = one
                    sh22 = one
                    sflag = -one
                else:
                    sh21 = -one
                    sh12 = one
                    sflag = -one
                if abs(d2) <= rgamsq:
                    d2 = d2 * gam * gam
                    sh21 = sh21 / gam
                    sh22 = sh22 / gam
                else:
                    d2 = d2 / (gam * gam)
                    sh21 = sh21 * gam
                    sh22 = sh22 * gam

    if sflag < zero:
        sparam[1] = sh11
        sparam[2] = sh21
        sparam[3] = sh12
        sparam[4] = sh22
    elif sflag == zero:
        sparam[2] = sh21
        sparam[3] = sh12
    else:
        sparam[1] = sh11
        sparam[4] = sh22

    sparam[0] = sflag
    return d1, d2, x1, sparam


def _mask_unused_sparam_torch(sparam: torch.Tensor) -> torch.Tensor:
    # BLAS rotmg uses sflag to indicate which parameter entries are meaningful.
    # Unused entries are implementation-defined and should not be compared.
    masked = sparam.clone()
    sflag = float(masked.reshape(-1)[0].item())

    if sflag == 1.0:
        masked[2] = 0
        masked[3] = 0
    elif sflag == 0.0:
        masked[1] = 0
        masked[4] = 0
    elif sflag == -2.0:
        masked[1] = 0
        masked[2] = 0
        masked[3] = 0
        masked[4] = 0

    return masked


def _mask_unused_sparam_infinicore(sparam):
    sparam_torch = convert_infinicore_to_torch(sparam)
    masked_torch = _mask_unused_sparam_torch(sparam_torch)
    sparam.copy_(infinicore_tensor_from_torch(masked_torch))


def parse_test_cases():
    test_cases = []
    for d1_0, d2_0, x1_0, y1_0 in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-5})
            d1_spec = TensorSpec.from_tensor((1,), None, dtype)
            d2_spec = TensorSpec.from_tensor((1,), None, dtype)
            x1_spec = TensorSpec.from_tensor((1,), None, dtype)
            y1_spec = TensorSpec.from_tensor((1,), None, dtype)
            out_d1_spec = TensorSpec.from_tensor((1,), None, dtype)
            out_d2_spec = TensorSpec.from_tensor((1,), None, dtype)
            out_x1_spec = TensorSpec.from_tensor((1,), None, dtype)
            out_param_spec = TensorSpec.from_tensor((5,), None, dtype)

            test_cases.append(
                TestCase(
                    inputs=[d1_spec, d2_spec, x1_spec, y1_spec],
                    kwargs={},
                    output_count=4,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"Rotmg - OUT_OF_PLACE (d1={d1_0}, d2={d2_0}, x1={x1_0}, y1={y1_0})",
                )
            )

            test_cases.append(
                TestCase(
                    inputs=[d1_spec, d2_spec, x1_spec, y1_spec],
                    kwargs={},
                    output_specs=[out_d1_spec, out_d2_spec, out_x1_spec, out_param_spec],
                    comparison_target="out",
                    tolerance=tol,
                    output_count=4,
                    description=f"Rotmg - INPLACE(out) (d1={d1_0}, d2={d2_0}, x1={x1_0}, y1={y1_0})",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Rotmg")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, d1, d2, x1, y1, **kwargs):
        out = kwargs.pop("out", None)
        out_d1, out_d2, out_x1, out_param = torch_rotmg(
            d1.clone().item(), d2.clone().item(), x1.clone().item(), y1.clone().item()
        )
        out_d1 = torch.tensor([out_d1], dtype=d1.dtype, device=d1.device)
        out_d2 = torch.tensor([out_d2], dtype=d2.dtype, device=d2.device)
        out_x1 = torch.tensor([out_x1], dtype=x1.dtype, device=x1.device)
        out_param = torch.tensor(out_param, dtype=d1.dtype, device=d1.device)
        if out is not None:
            out[0].copy_(out_d1)
            out[1].copy_(out_d2)
            out[2].copy_(out_x1)
            out[3].copy_(out_param)
            return out
        return out_d1, out_d2, out_x1, out_param

    def infinicore_operator(self, *args, **kwargs):
        result = infinicore.rotmg(*args, **kwargs)

        out = kwargs.get("out")
        if out is not None:
            _mask_unused_sparam_infinicore(out[3])
        else:
            _mask_unused_sparam_infinicore(result[3])

        return result


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()