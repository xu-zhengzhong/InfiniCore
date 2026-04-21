import torch
import ctypes
from ctypes import c_uint64, c_float, c_double
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

_TEST_CASES = [
    ((13,), None, None, 0.8, 0.6),
    ((13,), (10,), (10,), 0.8, 0.6),
    ((5632,), None, None, 0.9238795, 0.38268343),
    ((5632,), (5,), (5,), 0.9238795, 0.38268343),
    ((16,), (4,), (4,), 0.9659258, 0.25881904),
    ((5632,), (32,), (32,), 0.9659258, 0.25881904),
]

_TENSOR_DTYPES = [
    InfiniDtype.F32,
    # InfiniDtype.F64,
]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.F64: {"atol": 1e-12, "rtol": 1e-12},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_rot(x, y, c, s):
    x0 = x.clone()
    y0 = y.clone()
    x.copy_(c * x0 + s * y0)
    y.copy_(c * y0 - s * x0)


def _scalar_holders(c, s, dtype):
    if dtype == InfiniDtype.F64:
        return c_double(c), c_double(s)
    return c_float(c), c_float(s)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    c=0.8,
    s=0.6,
    dtype=torch.float32,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)
    y = TestTensor(shape, y_stride, dtype, device)

    if x.is_broadcast() or y.is_broadcast():
        return

    print(
        f"Testing Rot on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"y_stride:{y_stride} c:{c} s:{s} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch_rot(x.torch_tensor(), y.torch_tensor(), c, s)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRotDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            y.descriptor,
        )
    )

    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRotWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    c_holder, s_holder = _scalar_holders(c, s, dtype)

    def lib_rot():
        check_error(
            LIBINFINIOP.infiniopRot(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                y.data(),
                ctypes.byref(c_holder),
                ctypes.byref(s_holder),
                None,
            )
        )

    lib_rot()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch_rot(x.torch_tensor(), y.torch_tensor(), c, s),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation("    lib", lambda: lib_rot(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyRotDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")