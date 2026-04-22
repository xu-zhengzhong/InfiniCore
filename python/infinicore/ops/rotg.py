from infinicore.lib import _infinicore


def rotg(a, b, c, s):
    _infinicore.rotg_(
        a._underlying,
        b._underlying,
        c._underlying,
        s._underlying,
    )

    return a, b, c, s