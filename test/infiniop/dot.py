import torch
import ctypes
import math
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

# ==============================================================================
# Configuration
# ==============================================================================
# Format: (shape, x_stride, y_stride)
_TEST_CASES = [
    ((13,), None, None),
    ((13,), (10,), (10,)),
    ((5632,), None, None),
    ((5632,), (5,), (5,)),
    ((16,), (4,), (4,)),
    ((5632,), (32,), (32,)),
]

_TENSOR_DTYPES = [
    InfiniDtype.F32,
    InfiniDtype.F64,
]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F64: {"atol": 1e-7, "rtol": 1e-7},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_dot(x, y):
    return torch.dot(x, y)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    dtype=torch.float16,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)
    y = TestTensor(shape, y_stride, dtype, device)

    if dtype is InfiniDtype.F32:
        result = c_float(0.0)
    else:
        result = c_double(0.0)

    print(
        f"Testing Dot on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    expected_val = torch_dot(x.torch_tensor(), y.torch_tensor()).item()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDotDescriptor(
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
        LIBINFINIOP.infiniopGetDotWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_dot():
        check_error(
            LIBINFINIOP.infiniopDot(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                y.data(),
                ctypes.byref(result),
                None,
            )
        )

    lib_dot()

    actual_val = result.value
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        print(f"Expected Sum: {expected_val:.6f}, Actual Sum: {actual_val:.6f}")

    assert math.isclose(actual_val, expected_val, rel_tol=rtol, abs_tol=atol), \
        f"Value mismatch: actual={actual_val} != expected={expected_val} (atol={atol}, rtol={rtol})"

    if PROFILE:
        profile_operation("PyTorch", lambda: torch_dot(x.torch_tensor(), y.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_dot(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyDotDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")