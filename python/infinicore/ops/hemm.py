from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hemm(
    a: Tensor,
    b: Tensor,
    alpha: Tensor,
    beta: Tensor,
    out: Tensor,
    *,
    side: int = 0,
    uplo: int = 0,
):
    _infinicore.hemm_(
        a._underlying,
        b._underlying,
        alpha._underlying,
        beta._underlying,
        out._underlying,
        side,
        uplo,
    )

    return out
