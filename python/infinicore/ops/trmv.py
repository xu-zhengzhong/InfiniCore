from infinicore.lib import _infinicore


def trmv(a, x, *, uplo: int = 0, trans: int = 0, diag: int = 0):
    _infinicore.trmv_(
        a._underlying,
        x._underlying,
        uplo,
        trans,
        diag,
    )
    return x
