import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rot90(input: Tensor, k=1, dims=(0, 1)):
    r"""Rotate an n-D tensor by 90 degrees in the plane specified by dims axis.
    
    Rotation direction is from the first towards the second axis if k > 0, 
    and from the second towards the first for k < 0.
    
    Args:
        input (Tensor): the input tensor.
        k (int): number of times to rotate. Default value is 1.
        dims (tuple or list): axis to rotate. Default value is (0, 1).
        
    Returns:
        Tensor: The rotated tensor.
        
    Example:
        >>> x = torch.arange(4).view(2, 2)
        >>> x
        tensor([[0, 1],
                [2, 3]])
        >>> torch.rot90(x, 1, [0, 1])
        tensor([[1, 3],
                [0, 2]])
    """
    
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.rot90(input, k, dims)
    
    # Convert dims to list if it's a tuple
    dims_list = list(dims) if isinstance(dims, (tuple, list)) else [dims]
    
    res = _infinicore.rot90(input._underlying, k, dims_list)
    return Tensor(res)