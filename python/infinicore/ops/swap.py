from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def swap(input, other, *, out=None):
    if out is None:
        out_x, out_y = _infinicore.swap(input._underlying, other._underlying)
        return Tensor(out_x), Tensor(out_y)

    out_x, out_y = out
    _infinicore.swap_(
        input._underlying,
        other._underlying,
        out_x._underlying,
        out_y._underlying,
    )
    return out
