from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rotm(x, y, param, *, out=None):
    if out is None:
        out_x, out_y = _infinicore.rotm(x._underlying, y._underlying, param._underlying)
        return Tensor(out_x), Tensor(out_y)

    out_x, out_y = out
    _infinicore.rotm_(
        x._underlying,
        y._underlying,
        param._underlying,
        out_x._underlying,
        out_y._underlying,
    )
    return out