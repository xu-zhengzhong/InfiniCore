from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def axpby(x: Tensor, y: Tensor, *, alpha=1.0, beta=1.0, out=None):
    if out is None:
        return Tensor(_infinicore.axpby(x._underlying, y._underlying, alpha, beta))

    out.copy_(y)
    _infinicore.axpby_(x._underlying, out._underlying, alpha, beta)
    return out
