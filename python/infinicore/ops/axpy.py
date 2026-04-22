from infinicore.lib import _infinicore


def axpy(y, x, alpha):
    _infinicore.axpy_(y._underlying, x._underlying, alpha._underlying)

    return y
