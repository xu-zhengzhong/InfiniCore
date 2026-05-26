import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

_TEST_CASES_DATA = [
    (6, [0, 2, 5]),
    (8, [1, 3, 4, 7]),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [
    # infinicore.float16,
    # infinicore.bfloat16,
    infinicore.float32,
]


def parse_test_cases():
    test_cases = []
    for size, indices in _TEST_CASES_DATA:
        nnz = len(indices)
        for dtype in _TENSOR_DTYPES:
            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "size": size,
                        "indices": indices,
                    },
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpVV - OUT_OF_PLACE",
                )
            )
            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor((nnz,), dtype=dtype, name="values"),
                        TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "size": size,
                        "indices": indices,
                        "out": TensorSpec.from_tensor((), dtype=dtype, name="out"),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpVV - OUT(out)",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpVV")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, values, x, *, size, indices, out=None):
        sparse_dense = torch.zeros(size, dtype=values.dtype, device=values.device)
        sparse_dense[torch.tensor(indices, dtype=torch.int64, device=values.device)] = values
        result = torch.dot(sparse_dense, x)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, values, x, *, size, indices, out=None):
        device = values.device
        indices_tensor = infinicore.from_list(indices, dtype=infinicore.int64, device=device)
        sparse = infinicore.coo_spvec(indices_tensor, values, size)
        return infinicore.spvv(sparse, x, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
