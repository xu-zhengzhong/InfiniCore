from ..lib import _infinicore


def herk(a, alpha, beta, out, *, uplo=0, trans=0):
    _infinicore.herk_(
        a._underlying,
        alpha._underlying,
        beta._underlying,
        out._underlying,
        uplo,
        trans,
    )
    return out
