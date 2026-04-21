from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rotmg(d1, d2, x1, y1, *, out=None):
    if out is None:
        out_d1, out_d2, out_x1, out_param = _infinicore.rotmg(
            d1._underlying,
            d2._underlying,
            x1._underlying,
            y1._underlying,
        )
        return Tensor(out_d1), Tensor(out_d2), Tensor(out_x1), Tensor(out_param)

    out_d1, out_d2, out_x1, out_param = out
    _infinicore.rotmg_(
        d1._underlying,
        d2._underlying,
        x1._underlying,
        y1._underlying,
        out_d1._underlying,
        out_d2._underlying,
        out_x1._underlying,
        out_param._underlying,
    )
    return out