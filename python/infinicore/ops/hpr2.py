from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hpr2(
    alpha: Tensor,
    x: Tensor,
    y: Tensor,
    out: Tensor,
    *,
    uplo: int = 0,
):
    _infinicore.hpr2_(
        alpha._underlying,
        x._underlying,
        y._underlying,
        out._underlying,
        uplo,
    )

    return out
