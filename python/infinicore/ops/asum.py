from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def asum(input):
    return Tensor(_infinicore.asum(input._underlying))
