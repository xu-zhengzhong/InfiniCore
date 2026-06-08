import ctypes
from ctypes import c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    test_operator,
)

_TEST_CASES = [
    # n, x_stride, y_stride, alpha, beta
    (3, None, None, 1.0, 0.0),
    (257, (2,), None, 0.5, 1.0),
    (4096, None, (2,), -1.25, 0.25),
]

_TENSOR_DTYPES = [InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False


def test(
    handle,
    device,
    n,
    x_stride=None,
    y_stride=None,
    alpha=1.0,
    beta=1.0,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing Axpby on {InfiniDeviceNames[device]} with n:{n}, "
        f"alpha:{alpha}, beta:{beta}, dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)
    ans = alpha * x.torch_tensor() + beta * y.torch_tensor()
    y.set_tensor(y.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAxpbyDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            y.descriptor,
        )
    )

    for tensor in [x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAxpbyWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopAxpby(
            descriptor,
            workspace.data(),
            workspace_size.value,
            x.data(),
            y.data(),
            alpha,
            beta,
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    check_error(LIBINFINIOP.infiniopDestroyAxpbyDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
