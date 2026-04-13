from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def dot(input, other):
    return Tensor(_infinicore.dot(input._underlying, other._underlying))
