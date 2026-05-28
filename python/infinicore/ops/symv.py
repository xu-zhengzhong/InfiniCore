from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def symv(
    alpha: Tensor,
    a: Tensor,
    x: Tensor,
    beta: Tensor,
    out: Tensor,
    *,
    uplo: int = 0,
):
    _infinicore.symv_(
        alpha._underlying,
        a._underlying,
        x._underlying,
        beta._underlying,
        out._underlying,
        uplo,
    )

    return out
