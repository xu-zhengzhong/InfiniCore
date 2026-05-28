from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def trmm(
    a: Tensor,
    alpha: Tensor,
    b: Tensor,
    *,
    side: int = 0,
    uplo: int = 0,
    trans: int = 0,
    diag: int = 0,
):
    _infinicore.trmm_(
        a._underlying,
        alpha._underlying,
        b._underlying,
        side,
        uplo,
        trans,
        diag,
    )

    return b
