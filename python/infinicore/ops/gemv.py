from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def gemv(
    alpha: Tensor,
    a: Tensor,
    x: Tensor,
    beta: Tensor,
    out: Tensor,
    *,
    trans: int = 0,
):
    _infinicore.gemv_(
        alpha._underlying,
        a._underlying,
        x._underlying,
        beta._underlying,
        out._underlying,
        trans,
    )

    return out
