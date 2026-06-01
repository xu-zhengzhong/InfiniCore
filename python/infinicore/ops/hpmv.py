from infinicore.lib import _infinicore


def hpmv(alpha, ap, x, beta, out, *, uplo: int = 0):
    _infinicore.hpmv_(
        alpha._underlying,
        ap._underlying,
        x._underlying,
        beta._underlying,
        out._underlying,
        uplo,
    )
    return out
