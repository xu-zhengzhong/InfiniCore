import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def sgn(input: Tensor, *, out=None):
    r"""
    This function is an extension of sign() to complex tensors.
    It computes a new tensor whose elements have the same angles
    as the corresponding elements of the input tensor and absolute 
    values (i.e. magnitudes) of one for complex tensors and is 
    equivalent to sign() for non-complex tensors.
    
    Args:
        input (Tensor): the input tensor.
        
    Keyword Arguments:
        out (Tensor, optional): the output tensor.
        
    Returns:
        Tensor: A tensor with the signs of the input elements.
    """
    
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.sgn(input, out=out)
    
    if out is None:
        res = _infinicore.sgn(input._underlying)
        return Tensor(res)
    else:
        _infinicore.sgn_(input._underlying, out._underlying)
        return out