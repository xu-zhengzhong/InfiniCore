import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def quantile(input: Tensor, q, dim=None, keepdim=False, *, interpolation='linear', out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.quantile(input, q, dim=dim, keepdim=keepdim, 
                                               interpolation=interpolation, out=out)
    
    # Convert q to tensor if it's a scalar
    if isinstance(q, (int, float)):
        q_tensor = infinicore.tensor.from_list([q], dtype=infinicore.float32, device=input.device)
        is_scalar = True
    elif isinstance(q, Tensor):
        q_tensor = q
        is_scalar = False
    else:
        # Assume it's a list or other sequence
        q_tensor = infinicore.tensor.from_list(q, dtype=infinicore.float32, device=input.device)
        is_scalar = False

    # Validate interpolation mode
    valid_modes = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
    if interpolation not in valid_modes:
        raise ValueError(f"interpolation must be one of {valid_modes}, got '{interpolation}'")
    
    if out is None:
        res = _infinicore.quantile(input._underlying, q_tensor._underlying, 
                                   dim, keepdim, interpolation)
        result = Tensor(res)
        
        # If q was a scalar, squeeze the first dimension
        if is_scalar:
            result = result.squeeze(0)
        
        return result
    else:
        out_adjust = out.unsqueeze(0) if is_scalar else out
        _infinicore.quantile_(input._underlying, q_tensor._underlying, out_adjust._underlying, 
                             dim, keepdim, interpolation)
        return out