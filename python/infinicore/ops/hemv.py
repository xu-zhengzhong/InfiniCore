from infinicore.lib import _infinicore


def hemv(alpha, a, x, beta, out, *, uplo: int = 0):
    _infinicore.hemv_(
        alpha._underlying,
        a._underlying,
        x._underlying,
        beta._underlying,
        out._underlying,
        uplo,
    )
    return out
