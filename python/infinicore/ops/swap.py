from infinicore.lib import _infinicore


def swap(x, y):
    _infinicore.swap_(
        x._underlying,
        y._underlying,
    )

    return x, y
