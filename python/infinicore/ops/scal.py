from infinicore.lib import _infinicore


def scal(x, alpha):
    _infinicore.scal_(x._underlying, alpha)

    return x
