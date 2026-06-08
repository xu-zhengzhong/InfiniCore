import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from framework.utils.tensor_utils import infinicore_tensor_from_torch

_TEST_CASES_DATA = [
    # rows, cols, k, crow, col, alpha, beta
    (3, 4, 2, [0, 2, 3, 5], [0, 2, 1, 0, 3], 1.0, 0.0),
    (4, 5, 3, [0, 1, 1, 3, 4], [2, 0, 4, 1], 0.5, 1.0),
]

_TENSOR_DTYPES = [infinicore.float32]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def sampled_mm(a, b, values, rows, crow, col, alpha, beta):
    mm = torch.matmul(a, b)
    result = values.clone()
    for row in range(rows):
        for ptr in range(crow[row], crow[row + 1]):
            result[ptr] = alpha * mm[row, col[ptr]] + beta * values[ptr]
    return result


class SparseTestCase(TestCase):
    def __str__(self):
        nnz = len(self.kwargs["col"])
        return (
            f"TestCase({self.description} - "
            f"rows={self.kwargs['rows']}; cols={self.kwargs['cols']}; "
            f"k={self.kwargs['k']}; nnz={nnz}; alpha={self.kwargs['alpha']}; "
            f"beta={self.kwargs['beta']})"
        )


class CachedTensorSpec(TensorSpec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    @classmethod
    def from_tensor(cls, shape, strides=None, dtype=None, init_mode=None, **kwargs):
        if init_mode is None:
            return cls(shape=shape, dtype=dtype, strides=strides, **kwargs)
        return cls(
            shape=shape, dtype=dtype, strides=strides, init_mode=init_mode, **kwargs
        )

    def create_torch_tensor(self, device):
        if device not in self._cache:
            self._cache[device] = super().create_torch_tensor(device)
        return self._cache[device]


class CsrSpMatSpec(TensorSpec):
    def __init__(self, *, values_spec, rows, cols, crow, col, name="sparse"):
        super().__init__(shape=(rows, cols), dtype=values_spec.dtype, name=name)
        self.values_spec = values_spec
        self.rows = rows
        self.cols = cols
        self.crow = crow
        self.col = col
        self._cached_values = {}

    def create_torch_tensor(self, device):
        if device not in self._cached_values:
            self._cached_values[device] = self.values_spec.create_torch_tensor(
                device
            ).clone()
        values = self._cached_values[device]
        infini_values = infinicore_tensor_from_torch(values)
        infini_device = infini_values.device
        crow_tensor = infinicore.from_list(
            self.crow, dtype=infinicore.int64, device=infini_device
        )
        col_tensor = infinicore.from_list(
            self.col, dtype=infinicore.int64, device=infini_device
        )
        return infinicore.csr_spmat(
            crow_tensor, col_tensor, infini_values, (self.rows, self.cols)
        )

    def __str__(self):
        nnz = len(self.col)
        density = nnz / (self.rows * self.cols) if self.rows and self.cols else 0
        return f"{self.name}: spmat(rows={self.rows}, cols={self.cols}, nnz={nnz}, density={density:.6f})"


def parse_test_cases():
    test_cases = []
    for rows, cols, k, crow, col, alpha, beta in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
            values_spec = CachedTensorSpec.from_tensor(
                (nnz,), dtype=dtype, name="values"
            )
            test_cases.append(
                SparseTestCase(
                    inputs=[
                        values_spec,
                        CsrSpMatSpec(
                            values_spec=values_spec,
                            rows=rows,
                            cols=cols,
                            crow=crow,
                            col=col,
                        ),
                        TensorSpec.from_tensor((rows, k), dtype=dtype, name="a"),
                        TensorSpec.from_tensor((k, cols), dtype=dtype, name="b"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "k": k,
                        "crow": crow,
                        "col": col,
                        "alpha": alpha,
                        "beta": beta,
                    },
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SDDMM - INPLACE",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SDDMM")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self, values, sparse, a, b, *, rows, cols, k, crow, col, alpha, beta
    ):
        del sparse
        del cols, k
        return sampled_mm(a, b, values, rows, crow, col, alpha, beta)

    def infinicore_operator(
        self, values, sparse, a, b, *, rows, cols, k, crow, col, alpha, beta
    ):
        return infinicore.sddmm(sparse, a, b, alpha=alpha, beta=beta).values


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
