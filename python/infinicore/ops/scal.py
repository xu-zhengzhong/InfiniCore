from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def scal(input, alpha, *, out=None):
    if out is None:
        return Tensor(_infinicore.scal(input._underlying, alpha))

    _infinicore.scal_(out._underlying, input._underlying, alpha)
    return out
