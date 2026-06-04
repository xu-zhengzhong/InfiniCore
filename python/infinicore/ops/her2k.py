from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def her2k(
    a: Tensor,
    b: Tensor,
    alpha: Tensor,
    beta: Tensor,
    out: Tensor,
    *,
    uplo: int = 0,
    trans: int = 0,
):
    _infinicore.her2k_(
        a._underlying,
        b._underlying,
        alpha._underlying,
        beta._underlying,
        out._underlying,
        uplo,
        trans,
    )

    return out
