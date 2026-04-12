from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def axpy(input, other, alpha, *, out=None):
    if out is None:
        return Tensor(_infinicore.axpy(input._underlying, other._underlying, alpha._underlying))

    _infinicore.axpy_(input._underlying, other._underlying, alpha._underlying, out._underlying)

    return out
