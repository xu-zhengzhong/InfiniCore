from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rotg(a, b, *, out=None):
    if out is None:
        out_a, out_b, out_c, out_s = _infinicore.rotg(a._underlying, b._underlying)
        return Tensor(out_a), Tensor(out_b), Tensor(out_c), Tensor(out_s)

    out_a, out_b, out_c, out_s = out
    _infinicore.rotg_(
        a._underlying,
        b._underlying,
        out_a._underlying,
        out_b._underlying,
        out_c._underlying,
        out_s._underlying,
    )
    return out