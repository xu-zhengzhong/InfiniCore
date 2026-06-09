from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def blas_spmv(
    alpha: Tensor,
    ap: Tensor,
    x: Tensor,
    beta: Tensor,
    out: Tensor,
    *,
    uplo: int = 0,
):
    _infinicore.blas_spmv_(
        alpha._underlying,
        ap._underlying,
        x._underlying,
        beta._underlying,
        out._underlying,
        uplo,
    )

    return out
