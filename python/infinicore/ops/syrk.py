from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def syrk(
    a: Tensor,
    alpha: Tensor,
    beta: Tensor,
    out: Tensor,
    *,
    uplo: int = 0,
    trans: int = 0,
):
    _infinicore.syrk_(
        a._underlying,
        alpha._underlying,
        beta._underlying,
        out._underlying,
        uplo,
        trans,
    )

    return out
