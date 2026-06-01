from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hbmv(
    alpha: Tensor,
    a: Tensor,
    x: Tensor,
    beta: Tensor,
    out: Tensor,
    *,
    uplo: int = 0,
    k: int,
):
    _infinicore.hbmv_(
        alpha._underlying,
        a._underlying,
        x._underlying,
        beta._underlying,
        out._underlying,
        uplo,
        k,
    )

    return out
