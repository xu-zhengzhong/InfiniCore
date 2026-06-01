from infinicore.lib import _infinicore


def tpmv(ap, x, *, uplo: int = 0, trans: int = 0, diag: int = 0):
    _infinicore.tpmv_(
        ap._underlying,
        x._underlying,
        uplo,
        trans,
        diag,
    )
    return x
