from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def copy(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.copy(input._underlying, other._underlying))

    _infinicore.copy_(input._underlying, other._underlying, out._underlying)
    return out
