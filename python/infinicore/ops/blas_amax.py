from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def blas_amax(input):
    return Tensor(_infinicore.blas_amax(input._underlying))
