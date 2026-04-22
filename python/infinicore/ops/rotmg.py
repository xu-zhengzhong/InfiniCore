from infinicore.lib import _infinicore


def rotmg(d1, d2, x1, y1, param):
    _infinicore.rotmg_(
        d1._underlying,
        d2._underlying,
        x1._underlying,
        y1._underlying,
        param._underlying,
    )

    return d1, d2, x1, param