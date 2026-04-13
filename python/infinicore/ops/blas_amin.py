from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def blas_amin(input):
    return Tensor(_infinicore.blas_amin(input._underlying))
