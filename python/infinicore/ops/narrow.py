from typing import Optional
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def narrow(input: Tensor, dim: int, start: int, length: int, *, out: Optional[Tensor] = None) -> Tensor:
    r"""Returns a new tensor that is a narrowed version of input tensor. 
    
    The dimension dim is input from start to start + length.
    
    Args:
        input (Tensor): the tensor to narrow
        dim (int): the dimension along which to narrow
        start (int): the starting dimension
        length (int): the distance to the ending dimension
        out (Tensor, optional): the output tensor
        
    Returns: 
        Tensor:  Narrowed tensor
    """
    if not input.is_contiguous():
        input = input.contiguous()

    if out is not None:
        _infinicore.narrow_(out._underlying, input._underlying, dim, start, length)
        return out
    return Tensor(_infinicore.narrow(input._underlying, dim, start, length))