from infinicore.lib import _infinicore


def rotm(x, y, param):
    _infinicore.rotm_(x._underlying, y._underlying, param._underlying)

    return x, y