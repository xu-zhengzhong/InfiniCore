from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def log(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.log(input._underlying))
    _infinicore.log_(out._underlying, input._underlying)

    return out
