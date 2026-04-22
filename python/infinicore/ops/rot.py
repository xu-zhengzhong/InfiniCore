from infinicore.lib import _infinicore


def rot(x, y, c, s):
    _infinicore.rot_(x._underlying, y._underlying, c, s)

    return x, y