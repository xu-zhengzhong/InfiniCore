import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    CaseResult,
    GenericTestRunner,
    TensorSpec,
    TestCase,
    convert_infinicore_to_torch,
    infinicore_tensor_from_torch,
)

_TEST_CASES_DATA = [
    (3, 4, 2, [0, 2, 3, 5], [0, 2, 1, 0, 3]),
    (4, 5, 3, [0, 1, 1, 3, 4], [2, 0, 4, 1]),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for rows, cols, n, crow, col in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((cols, n), dtype=dtype, name="b"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "crow": crow,
                        "col": col,
                    },
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpMM - OUT_OF_PLACE",
                )
            )
            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((cols, n), dtype=dtype, name="b"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "crow": crow,
                        "col": col,
                        "out": TensorSpec.from_tensor(
                            (rows, n), dtype=dtype, name="out"
                        ),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpMM - OUT(out)",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpMM")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, values, b, *, rows, cols, crow, col, out=None):
        sparse = torch.sparse_csr_tensor(
            torch.tensor(crow, dtype=torch.int64, device=values.device),
            torch.tensor(col, dtype=torch.int64, device=values.device),
            values,
            size=(rows, cols),
        )
        result = torch.matmul(sparse, b)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, values, b, *, rows, cols, crow, col, out=None):
        device = values.device
        crow_tensor = infinicore.from_list(crow, dtype=infinicore.int64, device=device)
        col_tensor = infinicore.from_list(col, dtype=infinicore.int64, device=device)
        sparse = infinicore.csr_spmat(crow_tensor, col_tensor, values, (rows, cols))
        return infinicore.spmm(sparse, b, out=out)

    def run_test(self, device, test_case, config):
        values = test_case.inputs[0].create_torch_tensor(device)
        b = test_case.inputs[1].create_torch_tensor(device)
        kwargs = test_case.kwargs.copy()
        out_spec = kwargs.pop("out", None)
        torch_out = (
            out_spec.create_torch_tensor(device) if out_spec is not None else None
        )
        infini_out = (
            infinicore_tensor_from_torch(torch_out.clone())
            if torch_out is not None
            else None
        )

        try:
            torch_result = self.torch_operator(values, b, out=torch_out, **kwargs)
            infini_values = infinicore_tensor_from_torch(values)
            infini_b = infinicore_tensor_from_torch(b)
            infini_result = self.infinicore_operator(
                infini_values,
                infini_b,
                out=infini_out,
                **kwargs,
            )

            actual = convert_infinicore_to_torch(infini_result)
            expected = torch_result
            atol = test_case.tolerance["atol"]
            rtol = test_case.tolerance["rtol"]
            if expected.dtype == torch.bfloat16:
                actual = actual.to(torch.float32)
                expected = expected.to(torch.float32)
            assert torch.allclose(actual, expected, atol=atol, rtol=rtol)
            return CaseResult(
                success=True, return_code=0, test_case=test_case, device=device
            )
        except Exception as exc:
            return CaseResult(
                success=False,
                return_code=-1,
                error_message=str(exc),
                test_case=test_case,
                device=device,
            )


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
