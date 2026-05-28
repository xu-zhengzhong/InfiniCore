from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def tbmv(
    a: Tensor,
    x: Tensor,
    *,
    uplo: int = 0,
    trans: int = 0,
    diag: int = 0,
    k: int,
):
    _infinicore.tbmv_(
        a._underlying,
        x._underlying,
        uplo,
        trans,
        diag,
        k,
    )

    return x
