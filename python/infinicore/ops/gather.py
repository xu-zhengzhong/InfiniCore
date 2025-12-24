from typing import Optional
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def gather(input: Tensor, index: Tensor, *, dim: int, out: Optional[Tensor] = None) -> Tensor:
    r"""Gathers values along an axis specified by dim.

    This follows torch.gather semantics:
      - input and index must have the same number of dimensions
      - out has the same shape as index
    """

    # IMPORTANT: do NOT force input contiguous, because tests include strided inputs.
    # index in tests is contiguous already.

    if out is not None:
        _infinicore.gather_(out._underlying, input._underlying, index._underlying, dim)
        return out
    return Tensor(_infinicore.gather(input._underlying, index._underlying, dim))