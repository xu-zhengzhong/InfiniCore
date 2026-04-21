from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rot(input, other, c, s, *, out=None):
    if out is None:
        out_x, out_y = _infinicore.rot(input._underlying, other._underlying, c, s)
        return Tensor(out_x), Tensor(out_y)

    out_x, out_y = out
    _infinicore.rot_(
        input._underlying,
        other._underlying,
        out_x._underlying,
        out_y._underlying,
        c,
        s,
    )
    return out