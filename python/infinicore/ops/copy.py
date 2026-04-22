from infinicore.lib import _infinicore


def copy(x, y):
    _infinicore.copy_(x._underlying, y._underlying)

    return x
