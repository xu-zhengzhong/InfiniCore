from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def gemm(a: Tensor, b: Tensor, alpha: float, beta: float, out: Tensor):
    _infinicore.gemm_(
        out._underlying,
        a._underlying,
        b._underlying,
        float(alpha),
        float(beta),
    )

    return out
