from infinicore.lib import _infinicore


def hpr(alpha, x, ap, *, uplo: int = 0):
    _infinicore.hpr_(
        alpha._underlying,
        x._underlying,
        ap._underlying,
        uplo,
    )
    return ap
