from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def spr(
    alpha: Tensor,
    x: Tensor,
    out: Tensor,
    *,
    uplo: int = 0,
):
    _infinicore.spr_(
        alpha._underlying,
        x._underlying,
        out._underlying,
        uplo,
    )

    return out
