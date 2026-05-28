from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def ger(
    alpha: Tensor,
    x: Tensor,
    y: Tensor,
    out: Tensor,
):
    _infinicore.ger_(
        alpha._underlying,
        x._underlying,
        y._underlying,
        out._underlying,
    )

    return out
