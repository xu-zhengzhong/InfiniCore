from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def gbmv(
    alpha: Tensor,
    a: Tensor,
    x: Tensor,
    beta: Tensor,
    out: Tensor,
    *,
    trans: int = 0,
    kl: int,
    ku: int,
):
    _infinicore.gbmv_(
        alpha._underlying,
        a._underlying,
        x._underlying,
        beta._underlying,
        out._underlying,
        trans,
        kl,
        ku,
    )

    return out
