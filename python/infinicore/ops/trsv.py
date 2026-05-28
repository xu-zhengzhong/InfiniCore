from infinicore.lib import _infinicore


def trsv(a, x, *, uplo: int = 0, trans: int = 0, diag: int = 0):
    _infinicore.trsv_(
        a._underlying,
        x._underlying,
        uplo,
        trans,
        diag,
    )
    return x
