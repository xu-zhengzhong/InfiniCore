from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def her2(
    alpha: Tensor,
    x: Tensor,
    y: Tensor,
    out: Tensor,
    *,
    uplo: int = 0,
):
    _infinicore.her2_(
        alpha._underlying,
        x._underlying,
        y._underlying,
        out._underlying,
        uplo,
    )

    return out
