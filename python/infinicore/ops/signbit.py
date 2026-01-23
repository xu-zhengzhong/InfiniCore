import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def signbit(input: Tensor, *, out=None):
    r"""Tests if each element of input has its sign bit set or not.
    
    Args:
        input (Tensor): the input tensor.
        
    Keyword Arguments:
        out (Tensor, optional): the output tensor.
        
    Returns:
        Tensor: A boolean tensor indicating whether the sign bit is set.
    """
    
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.signbit(input, out=out)
    
    if out is None:
        res = _infinicore.signbit(input._underlying)
        return Tensor(res)
    else:
        _infinicore.signbit_(input._underlying, out._underlying)
        return out