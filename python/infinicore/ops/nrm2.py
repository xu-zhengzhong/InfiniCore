from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def nrm2(input):
    return Tensor(_infinicore.nrm2(input._underlying))
