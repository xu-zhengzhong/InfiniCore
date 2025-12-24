import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def log(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    r"""Apply the natural logarithm function, element-wise."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.log(input, inplace=inplace)

    if inplace:
        _infinicore.log_(input._underlying, input._underlying)
        return input

    if out is None:
        return Tensor(_infinicore.log(input._underlying))

    _infinicore.log_(out._underlying, input._underlying)

    return out
